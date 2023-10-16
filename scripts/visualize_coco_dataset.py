import os
import cv2
from glob import glob
import numpy as np
import argparse
import random
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi
from register_datasets import register_ycbv_dataset
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/home/ananth/datasets/ycbv", help="path to dataset")
    args = parser.parse_args()
    
    registered_datasets = register_ycbv_dataset(args.data_path)
    
    # print("Registered the following datasets: {}".format(registered_datasets))
    # print(DatasetCatalog.list())
    dataset_dicts = DatasetCatalog.get(registered_datasets[0])

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata={}, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ =="__main__":
    main()