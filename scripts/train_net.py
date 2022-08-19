import torch
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random, time, datetime
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_argument_parser, launch, HookBase
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader, DatasetMapper
from register_datasets import register_dopose_dataset, register_nvidia_fat_dataset, register_sps_dataset # registers dataset for detectron2. See register_datasets.py for more information
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import log_every_n_seconds
import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
import logging
setup_logger()
"""
TODO
- parse lr decaying?
- functionality to freeze specific layers
"""


def make_cfg(args, cfg_type="train", dataset=None):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # config file for mask r-cnn

    # Register all datasets
    if cfg_type == "train":
        _ = register_nvidia_fat_dataset(os.path.join(args.data_path, "fat"))
        _ = register_dopose_dataset(os.path.join(args.data_path, "doPose"))
        _ = register_sps_dataset(os.path.join(args.data_path, "sps_synthetic"))
    cfg.DATASETS.TRAIN = tuple(dataset)
    # add bitmask argument for COCO dataset with segmentations stored in RLE format (see: https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#standard-dataset-dicts)
    if args.dataset == "nvidia" or args.dataset == "dopose":
        cfg.INPUT.MASK_FORMAT = "bitmask"
    if args.dataset == "nvidia":
        cfg.DATASETS.TEST = ()
    elif args.dataset == "dopose":
        cfg.DATASETS.TEST = ()
    else:
        cfg.DATASETS.TEST = ()
    # cfg.TEST.EVAL_PERIOD = 500
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # instances are agnostic


    cfg.SOLVER.IMS_PER_BATCH = args.num_gpus * 4  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = args.lr  # pick a good LR
    cfg.SOLVER.WARMUP_ITERS = 100
    cfg.SOLVER.MAX_ITER = args.max_iter    
    if args.dataset == "dopose":
        cfg.SOLVER.STEPS = ()
    else:
        cfg.SOLVER.STEPS = ()
    cfg.SOLVER.CHECKPOINT_PERIOD = int(cfg.SOLVER.MAX_ITER / 5) # Checkpoint every 20%

    cfg.OUTPUT_DIR = "../output/"

    if cfg_type == "test":
        cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
        cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
        cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.05
        cfg.MODEL.DEVICE = 'cuda:0'
    
    if args.initial:
        print("Using base mask-rcnn model weights")
        cfg.MODEL.WEIGHTS = "../models/mask-rcnn/model_final_f10217.pkl"  # Baseline Mask-RCNN model
    else:
        print("Using weights from previous training iteration")
        cfg.MODEL.WEIGHTS = "../output/model_final.pth"

    return cfg

# custom class to include evaluator
class MyTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    # def build_hooks(self):
    #     hooks = super().build_hooks()
    #     hooks.insert(-1,LossEvalHook(
    #         self.cfg.TEST.EVAL_PERIOD,
    #         self.model,
    #         build_detection_test_loader(
    #             self.cfg,
    #             self.cfg.DATASETS.TEST[0],
    #             DatasetMapper(self.cfg,True)
    #         )
    #     ))
    #     return hooks

# class LossEvalHook(HookBase):
#     def __init__(self, eval_period, model, data_loader):
#         self._model = model
#         self._period = eval_period
#         self._data_loader = data_loader
    
#     def _do_loss_eval(self):
#         # Copying inference_on_dataset from evaluator.py
#         total = len(self._data_loader)
#         num_warmup = min(5, total - 1)
            
#         start_time = time.perf_counter()
#         total_compute_time = 0
#         losses = []
#         for idx, inputs in enumerate(self._data_loader):            
#             if idx == num_warmup:
#                 start_time = time.perf_counter()
#                 total_compute_time = 0
#             start_compute_time = time.perf_counter()
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#             total_compute_time += time.perf_counter() - start_compute_time
#             iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
#             seconds_per_img = total_compute_time / iters_after_start
#             if idx >= num_warmup * 2 or seconds_per_img > 5:
#                 total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
#                 eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
#                 log_every_n_seconds(
#                     logging.INFO,
#                     "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
#                         idx + 1, total, seconds_per_img, str(eta)
#                     ),
#                     n=5,
#                 )
#             loss_batch = self._get_loss(inputs)
#             losses.append(loss_batch)
#         mean_loss = np.mean(losses)
#         self.trainer.storage.put_scalar('validation_loss', mean_loss)
#         comm.synchronize()

#         return losses
            
#     def _get_loss(self, data):
#         # How loss is calculated on train_loop 
#         metrics_dict = self._model(data)
#         metrics_dict = {
#             k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
#             for k, v in metrics_dict.items()
#         }
#         total_losses_reduced = sum(loss for loss in metrics_dict.values())
#         return total_losses_reduced
        
        
#     def after_step(self):
#         next_iter = self.trainer.iter + 1
#         is_final = next_iter == self.trainer.max_iter
#         if is_final or (self._period > 0 and next_iter % self._period == 0):
#             self._do_loss_eval()
#         self.trainer.storage.put_scalars(timetest=12)

def main(args, dataset):
    cfg = make_cfg(args, cfg_type="train", dataset=dataset)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Visualize dataset
    if args.visualize:
        dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])

        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata={}, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow("", out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    # train network
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    trainer.train()

def parser_args():
    parser = default_argument_parser()
    parser.add_argument("--lr", type=float, default=0.00025, help="base learning rate")
    parser.add_argument("--max-iter", type=int, default=1700, help="max iterations for training")
    parser.add_argument("--visualize", action="store_true", help="visualize instances. Use for debug only")
    parser.add_argument("--data-path", type=str, default="/media/ws1/Data3/datasets/", help="parent dir of dataset for train/test")
    parser.add_argument("--dataset", type=str, default="nvidia", help="prefix for datasets when registering with detectron2")
    parser.add_argument("--confidence", type=float, default=0.7, help="confidence threshold for instance detection")
    parser.add_argument("--initial", action="store_true", help="uses base mask r-cnn instead of output from prev. training")
    parser.add_argument("--sps-path", type=str, default="../datasets/sps", help="path to sps dataset")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # clear cache to avoid orphaned cache
    torch.cuda.empty_cache()
    args = parser_args()
    if not args.eval_only:
        if args.dataset == "nvidia":
            # load chunks from nvidia_fat_chunks.json
            with open(os.path.join("nvidia_fat_chunks.json")) as file:
                chunks = json.load(file)
        elif args.dataset == "dopose":
            chunks = {}
            chunks["1"] = ["dopose_bin_train"]#, "dopose_table_train"]   
        elif args.dataset == "sps":
            chunks = {}
            chunks["1"] = ["sps_synthetic_train"] 
        chunk_keys = sorted(chunks.keys())
        for key in chunk_keys:
            launch(
                main,
                args.num_gpus,
                num_machines=args.num_machines,
                machine_rank=args.machine_rank,
                dist_url=args.dist_url,
                args=(args, chunks[key]),
            )
            args.initial = False
        
    # evaluate model and compare with baseline from IAS on sps dataset
    # register SPS test dataset. For now same as sps_train. Will be modified in the future.
    register_coco_instances("sps_val", {}, os.path.join(args.sps_path, "annotations/sps.json"), os.path.join(args.sps_path, "rgb"))
    cfg = make_cfg(args, cfg_type="test", dataset="sps_val")
    evaluator = COCOEvaluator("sps_val", ("segm",), False, output_dir = None)
    
    # Run metrics on baseline IAS
    cfg.MODEL.WEIGHTS = "../models/FAT_trained_Ml2R_bin_fine_tuned.pth"
    predictor = DefaultPredictor(cfg)
    val_loader = build_detection_test_loader(cfg, "sps_val")
    print("********IAS BASELINE MODEL********")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    # Run metrics on current model
    cfg.MODEL.WEIGHTS = "../output/model_final.pth"
    predictor = DefaultPredictor(cfg)
    val_loader = build_detection_test_loader(cfg, "sps_val")
    print("********CURRENT MODEL********")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    # display instance segmentations on random images from training set
    if args.visualize:
        cfg = make_cfg(args, cfg_type="test", dataset=chunks[key])
        predictor = DefaultPredictor(cfg)
        dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            predictions = predictor(img)
            visualizer = Visualizer(img[:, :, ::-1], metadata={}, scale=0.5)
            out = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))
            cv2.imshow("", out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    # clear cache to avoid orphaned cache
    torch.cuda.empty_cache()
