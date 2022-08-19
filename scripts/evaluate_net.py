#!/usr/bin/env python
import os
import numpy as np
import cv2
import argparse
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset;

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="path to class-agnostic model",
                        default='../models/FAT_trained_Ml2R_bin_fine_tuned.pth')
    args = parser.parse_args()

    confidence = 0.7

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # config file for mask r-cnn
    cfg.MODEL.WEIGHTS = args.model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.MODEL.DEVICE = 'cuda:0'

    # Evaluation section
    predictor = DefaultPredictor(cfg)
    json_path = "../datasets/sps/annotations/sps.json"
    imgs_path = "../datasets/sps/rgb"
    register_coco_instances("sps_val", {}, json_path, imgs_path)
    evaluator = COCOEvaluator("sps_val", ("segm",), False, output_dir = None)
    val_loader = build_detection_test_loader(cfg, "sps_val")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == '__main__':
    main()
