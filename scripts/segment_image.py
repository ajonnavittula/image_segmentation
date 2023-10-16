import os
import numpy as np
import cv2
import argparse
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import torch

torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, help="path rgb to image", default='../datasets/sps/rgb/0000.png')
    parser.add_argument("--model-path", type=str, help="path to class-agnostic model",
                        default='../models/FAT_trained_Ml2R_bin_fine_tuned.pth')
    parser.add_argument("--visualize", action="store_true", help="visualize instances")
    args = parser.parse_args()

    # get absolute path
    args.image_path = os.path.abspath(args.image_path)
    args.model_path = os.path.abspath(args.model_path)
    
    confidence = 0.1

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # config file for mask r-cnn
    cfg.MODEL.WEIGHTS = args.model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = True
    cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence
    cfg.MODEL.DEVICE = 'cuda:0'
    
    img = cv2.imread(args.image_path)
    predictor = DefaultPredictor(cfg)
    predictions = predictor(img)
    areas = predictions["instances"].pred_boxes.area()
    print(areas)
    visualizer = Visualizer(img[:, :, ::-1], metadata={}, scale=1.0)
    out = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))

    if args.visualize:
        cv2.imshow("", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
