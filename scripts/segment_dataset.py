#!/usr/bin/env python
import os
import numpy as np
import cv2
import argparse, time
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_argument_parser
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, help="path to dataset with rgb-d images",
                        default="../datasets/sps")
    parser.add_argument("--model-path", type=str, help="path to class-agnostic model",
                        default='../output/model_final.pth')
    parser.add_argument("--visualize", action="store_true", help="visualize instances")
    args = parser.parse_args()

    # Segment images from dataset
    args.model_path = os.path.abspath(args.model_path)
    rgb_img_path = os.path.abspath(args.data_path) + "/rgb"

    confidence = 0.9

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # config file for mask r-cnn
    cfg.MODEL.WEIGHTS = args.model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.05
    cfg.MODEL.DEVICE = 'cuda:0'

    predictor = DefaultPredictor(cfg)
    MetadataCatalog.get("user_data").set(thing_classes=[""])
    metadata = MetadataCatalog.get("user_data")
    for filename in os.listdir(rgb_img_path):
        img = cv2.imread(rgb_img_path + "/" + filename)
        predictions = predictor(img)
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata)
        out = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))
        seg_img = out.get_image()[:, :, ::-1]
        if args.visualize:
            cv2.imshow('segmented_image', seg_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        dataset_name = args.data_path.split("/")[-1]
        savedir = "../results/" + dataset_name + "/ias"
        os.makedirs(savedir, exist_ok=True)

        cv2.imwrite(savedir + "/" + filename, seg_img)


if __name__ == '__main__':
    main()
