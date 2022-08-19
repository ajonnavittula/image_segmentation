"""
This script creates a COCO format json from images and segmentations stored as bitmasks.
Expected Data
"""

import argparse
from distutils.log import INFO
import os
import cv2
import json
from glob import glob
from detectron2.structures import BoxMode
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi
from detectron2.data.datasets import register_coco_instances
import random
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer

# NVIDIA FAlling Things (FAT) dataset: https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation
class FAT():
    """
    Convert NVIDIA FAT dataset to COCO Json format.
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
    def __init__(self, args):
        self.info = {"year" : 2018,
                     "version": "1.0",
                     "description" : "A synthetic dataset for advancing the state-of-the-art in object detection and 3D pose estimation in the context of robotics.",
                     "contributor" : "Jonathan Tremblay, Thang To, Stan Birchfield",
                     "url" : "https://research.nvidia.com/publication/2018-06_falling-things-synthetic-dataset-3d-object-detection-and-pose-estimation",
                     "date_created" : "2022"
        }
        self.licenses = [{"1":0}]
        self.type = "instances"
        self.parent_dir = args.data_path
        self.categories = [{"id": 1, "name": "object", "supercategory": "object"}]
        self.cat2id = {cat["name"]: catId+1 for catId, cat in enumerate(self.categories)}
        self.annoId = 0
        self.visualize = args.visualize

        # Folders of interest
        subdirs = ["mixed", "single"]
        # find folders that have jsons already
        annotationpath = os.path.join(self.parent_dir, "annotations")
        os.makedirs(annotationpath, exist_ok=True)
        completed_jsons = [os.path.basename(f.path).replace(".json", "") for f in os.scandir(annotationpath)]

        for subdir in subdirs:
            subdir = os.path.join(self.parent_dir, subdir)
            for f in os.scandir(subdir):
                if f.is_dir():
                    datapath = f.path
                    # check if coco file already exists
                    name = os.path.basename(f.path)
                    if name in completed_jsons:
                        print("JSON already exists for {}. Skipping".format(name))
                        continue

                    dir = os.path.basename(datapath)
                    print("Working on folder: {}".format(dir))
                    img_list = []
                    for dir, _, _ in os.walk(datapath):
                        img_list.extend(glob(os.path.join(dir, "*.jpg")))
                    images = self.__get_image_annotation_pairs__(img_list)
                    json_data = {"info" : self.info,
                                    "images" : images,
                                    "licenses" : self.licenses,
                                    "type" : self.type,
                                    "annotations" : self.annotations,
                                    "categories" : self.categories}
                    
                    with open(os.path.join(annotationpath, os.path.basename(datapath) + ".json"), "w") as jsonfile:
                        json.dump(json_data, jsonfile, sort_keys=True, indent=None, separators=(',', ': '))

    
    def __get_image_annotation_pairs__(self, img_list):
        images = []
        self.annotations = []
        for imId, impath in enumerate(tqdm(img_list)):
            image = cv2.imread(impath)
            mask_path = impath.replace(".jpg", ".seg.png")
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
            if np.all(mask==0): #ignore imgs with no visible masks
                continue
            
            self.__get_annotation__(mask, image, imId)

            images.append({"date_captured" : "2016",
                           "file_name" : os.path.relpath(impath, self.parent_dir),
                           "id" : imId+1,
                           "license" : 1,
                           "url" : "",
                           "height" : mask.shape[0],
                           "width" : mask.shape[1]})
        return images
    def __get_annotation__(self, mask, image, imId):
        contours = None
        # Get instances
        instances = np.unique(mask)
        if instances[0] == 0: #avoid unlabeled areas
            instances = instances[1:]

        for instance in instances:
            mask_instance = mask == instance
            mask_instance = np.array(mask_instance).astype(np.uint8)
            contours, _ = cv2.findContours(mask_instance, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            segmentation = []
            for contour in contours:
                if contour.size >= 6:
                    segmentation.append(contour.flatten().tolist())
            if not segmentation:
                continue
            [x, y, w, h] = cv2.boundingRect(mask_instance)
            RLEs = cocomask.frPyObjects(segmentation, mask_instance.shape[0], mask_instance.shape[1])
            RLE = cocomask.merge(RLEs)
            area = cocomask.area(RLE)
            # segmentation = cocomask.encode(np.asfortranarray(mask))
            # segmentation["counts"] = segmentation["counts"].decode("utf-8")
            # area = cocomask.area(segmentation).item()
            # bbox =  cocomask.toBbox(segmentation).astype(int).tolist()
            self.annotations.append({"segmentation" : segmentation, 
                                "area" : np.float(area),
                                "iscrowd" : 0,
                                "image_id" : imId+1,
                                # "bbox": bbox,
                                "bbox" : [x, y, w, h],
                                # "bbox_mode": BoxMode.XYWH_ABS,
                                "category_id" : 1, # only one category
                                "id": self.annoId+1})
            self.annoId += 1

class DoPose():
    """
    Create usable COCO json for DoPose dataset.
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
    def __init__(self, args):
        self.info = {"year" : 2022,
                     "version": "1.0",
                     "description" : "A dataset of highly cluttered and closely stacked objects.",
                     "contributor" : " Gouda, Anas; Nedungadi, Ashwin; Ghatpande, Anay;  Reining, Christopher;  Youssef, Hazem; Roidl, Moritz",
                     "url" : "https://zenodo.org/record/6103779#.YtHUytLMJhE",
                     "date_created" : "2022"
        }
        self.licenses = [{"1":0}]
        self.type = "instances"
        self.datapath = args.data_path
        self.categories = [{"id": 1, "name": "object", "supercategory": "object"}]
        self.cat2id = {cat["name"]: catId+1 for catId, cat in enumerate(self.categories)}
        self.annoId = 0
        self.visualize = args.visualize

        img_list = []
        folders = ["test_bin", "test_table"]
        for folder in folders:
            dir = os.path.join(self.datapath, folder)
            subfolders = os.listdir(dir)
            for subfolder in subfolders:
                img_list.extend(glob(os.path.join(dir, subfolder, "rgb/*.png")))
            print("Found {} images in {} directory".format(len(img_list), folder))
            images = self.__get_image_annotation_pairs__(img_list)
            json_data = {"info" : self.info,
                            "images" : images,
                            "licenses" : self.licenses,
                            "type" : self.type,
                            "annotations" : self.annotations,
                            "categories" : self.categories}
            annotationpath = os.path.join(self.datapath, "annotations")
            os.makedirs(annotationpath, exist_ok=True)
            with open(os.path.join(annotationpath, folder + ".json"), "w") as jsonfile:
                json.dump(json_data, jsonfile, sort_keys=True, indent=None, separators=(',', ': '))

    def __get_image_annotation_pairs__(self, img_list):
        images = []
        self.annotations = []
        for imId, impath in enumerate(tqdm(img_list)):
            image = cv2.imread(impath)

            self.__get_annotation__(impath, image, imId)
            images.append({"date_captured" : "2022",
                        "file_name" : os.path.relpath(impath, self.datapath),
                        "id" : imId+1,
                        "license" : 1,
                        "url" : "",
                        "height" : image.shape[0],
                        "width" : image.shape[1]})
        return images

    def __get_annotation__(self, impath, image, imId):
            im_id = int(os.path.basename(impath).replace(".png", ""))
            folder = os.path.dirname(os.path.dirname(impath))
            with open(os.path.join(folder, "scene_gt_coco_modal.json")) as file:
                gt_coco = json.load(file)
            all_anns = gt_coco["annotations"]
            for ann in all_anns:
                if ann["image_id"] == im_id:
                    del ann["ignore"]
                    ann["image_id"] = imId+1
                    ann["category_id"] = 1
                    ann["id"] = self.annoId + 1
                    self.annotations.append(ann)
                    self.annoId += 1 

class SPS():
    """
    Convert ABB SPS dataset to COCO Json format.
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
    def __init__(self, args):
        self.info = {"year" : 2022,
                     "version": "1.0",
                     "description" : "A custom dataset for small package singulation (SPS)",
                     "contributor" : "",
                     "url" : "",
                     "date_created" : "2022"
        }
        self.licenses = [{"1":0}]
        self.type = "instances"
        self.parent_dir = args.data_path
        self.categories = [{"id": 1, "name": "object", "supercategory": "object"}]
        self.cat2id = {cat["name"]: catId+1 for catId, cat in enumerate(self.categories)}
        self.annoId = 0
        self.visualize = args.visualize

        # find folders that have jsons already
        annotationpath = os.path.join(self.parent_dir, "annotations")
        os.makedirs(annotationpath, exist_ok=True)
        datapath = os.path.join(self.parent_dir, "rgb")
        img_list = []
        for dir, _, _ in os.walk(datapath):
            img_list.extend(glob(os.path.join(dir, "*.png")))
        images = self.__get_image_annotation_pairs__(img_list)
        json_data = {"info" : self.info,
                        "images" : images,
                        "licenses" : self.licenses,
                        "type" : self.type,
                        "annotations" : self.annotations,
                        "categories" : self.categories}
        with open(os.path.join(annotationpath, os.path.basename(self.parent_dir) + ".json"), "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=None, separators=(',', ': '))
    
    def __get_image_annotation_pairs__(self, img_list):
        images = []
        self.annotations = []
        for imId, impath in enumerate(tqdm(img_list)):
            image = cv2.imread(impath)
            im_name = os.path.basename(impath)
            mask_path = os.path.join(self.parent_dir, "gt", im_name)
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
            if np.all(mask==0): #ignore imgs with no visible masks
                continue
            
            self.__get_annotation__(mask, image, imId)

            images.append({"date_captured" : "2022",
                           "file_name" : os.path.relpath(impath, self.parent_dir),
                           "id" : imId+1,
                           "license" : 1,
                           "url" : "",
                           "height" : mask.shape[0],
                           "width" : mask.shape[1]})
        return images

    def __get_annotation__(self, mask, image, imId):
        contours = None
        # Get instances
        instances = np.unique(mask)
        if instances[0] == 0: #avoid unlabeled areas
            instances = instances[1:]

        for instance in instances:
            # avoid blue background from seg images
            if instance == 14:
                continue
            mask_instance = mask == instance
            mask_instance = np.array(mask_instance).astype(np.uint8)
            contours, _ = cv2.findContours(mask_instance, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            segmentation = []
            for contour in contours:
                if contour.size >= 6:
                    segmentation.append(contour.flatten().tolist())
            if not segmentation:
                continue
            [x, y, w, h] = cv2.boundingRect(mask_instance)
            RLEs = cocomask.frPyObjects(segmentation, mask_instance.shape[0], mask_instance.shape[1])
            RLE = cocomask.merge(RLEs)
            area = cocomask.area(RLE)
            self.annotations.append({"segmentation" : segmentation, 
                                "area" : np.float(area),
                                "iscrowd" : 0,
                                "image_id" : imId+1,
                                "bbox" : [x, y, w, h],
                                "category_id" : 1, # only one category
                                "id": self.annoId+1})
            self.annoId += 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true", help="visualize instances. Use for debug only")
    parser.add_argument("--dataset", type=str, default="nvidia", help="choose which dataset to create coco json for")
    parser.add_argument("--data-path", type=str, default="/media/ws1/Data3/datasets/fat", help="path to dataset")
    args = parser.parse_args()
    
    if args.dataset == "nvidia":
        print("NVIDIA FAT dataset is large. Creating COCO JSON's for folders inside parent dir")
        FAT(args)
    elif args.dataset == "dopose":
        print("Creating COCO format JSON files for the doPose dataset.")
        DoPose(args)
    elif args.dataset == "sps":
        print("Creating COCO format JSON for SPS dataset")
        SPS(args)
    else:
        print("Unsupported dataset. The options are nvidia or dopose")

    if args.visualize:
        annotationpath = os.path.join(args.data_path, "annotations")
        annotations = random.sample(os.listdir(annotationpath), 1)[0]
        annotations = os.path.join(annotationpath, annotations)
        register_coco_instances(args.dataset + "_train", {}, annotations, args.data_path)
        dataset_dicts = DatasetCatalog.get(args.dataset + "_train")

        for d in random.sample(dataset_dicts, 3):
            img = cv2.imread(d["file_name"])
            visualizer = Visualizer(img[:, :, ::-1], metadata={}, scale=0.5)
            out = visualizer.draw_dataset_dict(d)
            cv2.imshow("", out.get_image()[:, :, ::-1])
            cv2.waitKey(0)
        cv2.destroyAllWindows()
            


if __name__ == "__main__":
    main()