#!/usr/bin/env python
import os
import numpy as np
import cv2
from skimage.io import imsave
import torch
import argparse
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_argument_parser
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

# Saves every instance mask individually
def output_masks():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", type=str, help="path rgb to image", default='../datasets/sps/rgb/')
    parser.add_argument("--model-path", type=str, help="path to class-agnostic model",
                        default='../models/FAT_trained_Ml2R_bin_fine_tuned.pth')
    args = parser.parse_args()

    # get absolute path
    args.image_path = os.path.abspath(args.image_path)
    images_ids = next(os.walk([args.image_path][0]))[2]
    args.model_path = os.path.abspath(args.model_path)
    msk_path = "../datasets/sps/masks/"

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

    for j in range(len(images_ids)):
        img = cv2.imread(args.image_path + "/" + images_ids[j])
        predictor = DefaultPredictor(cfg)
        predictions = predictor(img)

        for i in range(predictions["instances"].pred_masks.shape[0]):
            mask0 = predictions["instances"].pred_masks[i].cpu().numpy()*255.0
            mask0 = mask0.astype(np.uint8)
            mask_path = msk_path + images_ids[j][:-4] + "_" + str(i).zfill(4) + ".png"
            imsave(mask_path, mask0)
            # cv2.imshow("",mask0)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

if __name__ == '__main__':
    output_masks()