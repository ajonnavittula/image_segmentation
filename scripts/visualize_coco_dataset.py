import os
import cv2
from glob import glob
import numpy as np
import argparse
import random
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="/media/ws1/Data3/datasets/fat", help="path to dataset")
    args = parser.parse_args()

    annotationpath = os.path.join(args.data_path, "annotations")
    annotations = random.sample(os.listdir(annotationpath), 1)[0]
    annotations = os.path.join(annotationpath, annotations)
    register_coco_instances("sample_train", {}, annotations, args.data_path)
    dataset_dicts = DatasetCatalog.get("sample_train")

    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata={}, scale=0.5)
        out = visualizer.draw_dataset_dict(d)
        cv2.imshow("", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ =="__main__":
    main()