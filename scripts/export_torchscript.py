#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
import onnx
import torch
from torch import Tensor

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.export import (
    Caffe2Tracer,
    TracingAdapter,
    add_export_config,
    dump_torchscript_IR,
    scripting_with_instances,
)
from detectron2.modeling import GeneralizedRCNN, RetinaNet, build_model
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.projects.point_rend import add_pointrend_config
from detectron2.structures import Boxes
from detectron2.utils.env import TORCH_VERSION
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import setup_logger
from register_datasets import register_sps_dataset
from detectron2.data import DatasetCatalog

def setup_cfg(args):
    cfg = get_cfg()
    # cuda context is initialized before creating dataloader, so we don't fork anymore
    cfg.DATASETS.TEST = register_sps_dataset(os.path.join(args.data_path, "sps_synthetic"))
    cfg.DATALOADER.NUM_WORKERS = 0
    add_pointrend_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # set num classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # instances are agnostic
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.05
    cfg.freeze()
    return cfg

# experimental. API not yet final
def export_tracing(torch_model, inputs):
    assert TORCH_VERSION >= (1, 8)
    image = inputs[0]["image"]
    inputs = [{"image": image}]  # remove other unused keys

    if isinstance(torch_model, GeneralizedRCNN):

        def inference(model, inputs):
            # use do_postprocess=False so it returns ROI mask
            inst = model.inference(inputs, do_postprocess=False)[0]
            return [{"instances": inst}]

    else:
        inference = None  # assume that we just call the model directly

    traceable_model = TracingAdapter(torch_model, inputs, inference)

    if args.format == "torchscript":
        ts_model = torch.jit.trace(traceable_model, (image,))
        with PathManager.open(os.path.join(args.output, "model.ts"), "wb") as f:
            torch.jit.save(ts_model, f)
        dump_torchscript_IR(ts_model, args.output)
    elif args.format == "onnx":
        # NOTE onnx export currently failing in pytorch
        with PathManager.open(os.path.join(args.output, "model.onnx"), "wb") as f:
            torch.onnx.export(traceable_model, (image,), f)
    logger.info("Inputs schema: " + str(traceable_model.inputs_schema))
    logger.info("Outputs schema: " + str(traceable_model.outputs_schema))

    if args.format != "torchscript":
        return None
    if not isinstance(torch_model, (GeneralizedRCNN, RetinaNet)):
        return None

    def eval_wrapper(inputs):
        """
        The exported model does not contain the final resize step, which is typically
        unused in deployment but needed for evaluation. We add it manually here.
        """
        input = inputs[0]
        instances = traceable_model.outputs_schema(ts_model(input["image"]))[0]["instances"]
        postprocessed = detector_postprocess(instances, input["height"], input["width"])
        return [{"instances": postprocessed}]

    return eval_wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a model for deployment.")
    parser.add_argument(
        "--format",
        choices=["torchscript"],
        help="output format",
        default="torchscript",
    )
    parser.add_argument(
        "--export-method",
        choices=["tracing"],
        help="Method to export models",
        default="tracing",
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--sample-image", default="../datasets/sps/rgb/0000.png", type=str, help="sample image for input")
    parser.add_argument("--data-path", type=str, default="/media/ws1/Data3/datasets/", help="parent dir of dataset for train/test")
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--output", default = "../export", help="output directory for the converted model")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    logger = setup_logger()
    logger.info("Command line arguments: " + str(args))
    PathManager.mkdirs(args.output)
    # Disable respecialization on new shapes. Otherwise --run-eval will be slow
    torch._C._jit_set_bailout_depth(1)

    cfg = setup_cfg(args)

    # create a torch model
    torch_model = build_model(cfg)
    DetectionCheckpointer(torch_model).resume_or_load(cfg.MODEL.WEIGHTS)
    torch_model.eval()

    # get a sample data
    data_loader = build_detection_test_loader(cfg, "sps_synthetic_train")
    first_batch = next(iter(data_loader))

    # convert and save model
    exported_model = export_tracing(torch_model, first_batch)

    # run evaluation with the converted model
    # run evaluation with the converted model
    if args.run_eval:
        assert exported_model is not None, (
            "Python inference is not yet implemented for "
            f"export_method={args.export_method}, format={args.format}."
        )
        logger.info("Running evaluation ... this takes a long time if you export to CPU.")
        dataset = "sps_synthetic_train"
        data_loader = build_detection_test_loader(cfg, dataset)
        # NOTE: hard-coded evaluator. change to the evaluator for your dataset
        evaluator = COCOEvaluator(dataset, output_dir=args.output)
        metrics = inference_on_dataset(exported_model, data_loader, evaluator)
        print_csv_format(metrics)
    logger.info("Success.")
