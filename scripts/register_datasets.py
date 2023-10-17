"""
Code to register datasets to detectron2. 
Datasets have to be registered for each instance before they can be found by detectron2. 
More info here: https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
Tutorial used: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5
"""

import os
import cv2
from glob import glob
import numpy as np
from PIL import Image
from detectron2.structures import BoxMode
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from copy import copy
import json

""" 
Register COCO format datasets.
See create_coco_json.py for info on how to create COCO format dataset.
"""

# register_coco_instances("sps_dataset_train", {}, "/home/ws1/Downloads/sps.json","/home/ws1/Downloads/sps")
# register_coco_instances("nvidia_train", {}, "/media/ws1/Data3/datasets/fat/Annotations/036_wood_block_16k.json","/media/ws1/Data3/datasets/fat")

def register_ycbv_dataset(path):
    path = os.path.join(path, "train_pbr")
    registered_datasets = []
    for folder in os.scandir(path):
        dataset_name = "ycbv_" + os.path.basename(folder) + "_train"
        annotation = os.path.join(folder, "scene_gt_coco.json")
        if not dataset_name in DatasetCatalog.list():
            register_coco_instances(dataset_name, {}, annotation, folder)
            registered_datasets.append(dataset_name)
    return registered_datasets

def register_nvidia_fat_dataset(path):
    """
    Register nvidia FAT Dataset (https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation)
    The dataset must be organized as follows:
    path
    ├── annotations
    |             ├──folder1.json
    |             ├──folder2.json
    |             └──...
    |
    └── single # dataset with single object
        ├── single # dataset with single object
        │     ├──folder1
        |     |        └──...
        │     ├──folder2
        |     |        └──...
        │     └──...
        └── mixed # dataset with mixed objects
                └──...
    """
    annotationpath = os.path.join(path, "annotations")
    annotations = [f.path for f in os.scandir(annotationpath) if not f.is_dir()]
    registered_datasets = []
    for anno in annotations:
        dataset_name = os.path.basename(anno).replace(".json", "")
        dataset_name = "nvidia_fat_" + dataset_name + "_train"
        if not dataset_name in DatasetCatalog.list():
            register_coco_instances(dataset_name, {}, anno, path)
            registered_datasets.append(dataset_name)
    return registered_datasets

def register_dopose_dataset(path):
    """
    Register DoPose Dataset (https://zenodo.org/record/6103779#.YtGxxdLMJhE)
    The dataset must be organized as follows:
    path
    ├── test_bin
    |          ├── 000001
    |          |        └──...  
    |          ├── 000002        
    |          |        └──...           
    |          └── ...        
    └── test_table
                 └── ...   
    """
    annotationpath = os.path.join(path, "annotations")
    annotations = [f.path for f in os.scandir(annotationpath) if not f.is_dir()]
    registered_datasets = []
    for anno in annotations:
        dataset_name = os.path.basename(anno).replace(".json", "").split("_")[1]
        dataset_name = "dopose_" + dataset_name + "_train"
        if not dataset_name in DatasetCatalog.list():
            register_coco_instances(dataset_name, {}, anno, path)
            registered_datasets.append(dataset_name)
    return registered_datasets

def register_sps_dataset(path):
    """
    Register custom ABB SPS dataset.
    The dataset must be organized as follows:
    path
    ├── rgb
    |     ├── 000001.png
    |     ├── 000002.png        
    |     └── ...        
    └── gt
         ├── 000001.png
         ├── 000002.png        
         └── ...          
    """
    annotationpath = os.path.join(path, "annotations")
    annotations = [f.path for f in os.scandir(annotationpath) if not f.is_dir()]
    registered_datasets = []
    for anno in annotations:
        dataset_name = os.path.basename(anno).replace(".json", "")
        dataset_name = dataset_name + "_train"
        if not dataset_name in DatasetCatalog.list():
            register_coco_instances(dataset_name, {}, anno, path)
            registered_datasets.append(dataset_name)
    return registered_datasets

"""
Code to register non COCO format custom dataset. 
Deprecated and here for reference only.
"""
datapath = "/media/ws1/Data3/datasets/test"

def get_fat_dicts(datapath):
    pattern = "*.jpg"
    files = []
    for dir, _, _ in os.walk(datapath):
        files.extend(glob(os.path.join(dir, pattern)))
    print("start registration")
    dataset_dicts = []
    for idx, filename in enumerate(files):
        record = {}
        # remove file extension
        file = filename.replace(".jpg", "")
        height, width = cv2.imread(filename).shape[:2]

        record["filename"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []
        instance_img = cv2.imread(file + ".seg.png")
        instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2GRAY)
        # Get instances
        instances = np.unique(instance_img)
        if instances[0] == 0: #avoid unlabeled areas
            instances = instances[1:]

        for instance in instances:
            mask = instance_img == instance
            print(mask)
            py, px = np.where(mask==True) #bbox
            mask = cocomask.encode(mask.astype(np.uint8, order="F"))

            obj = {"bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": mask,
                    "category_id": 0
            }
            objs.append(obj)
        record["annotation"] = objs
        dataset_dicts.append(record)
    print("registration complete")
    return dataset_dicts

# uncomment to register dataset
# DatasetCatalog.register("fat_dataset_train", lambda d="train": get_fat_dicts(datapath))
# MetadataCatalog.get("fat_dataset_train").set(thing_classes=["objects"])
# fat_metadata = MetadataCatalog.get("fat_dataset_train")
# dataset_dicts = get_fat_dicts(datapath)



