import os
import numpy as np
import cv2
import argparse
import socket
import json
import shutil
import time
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

def get_predictions(predictor, img):
    pred_time = time.time()
    predictions = predictor(img)
    pred_time  = time.time() - pred_time
    # scores found to be deceiving
    # scores = predictions["instances"].scores.to("cpu").numpy()
    mask_array = predictions["instances"].pred_masks.to("cpu").numpy()
    num_instances = mask_array.shape[0]
    instance_colors = np.linspace(10, 255, num_instances).astype(np.uint8)
    mask_array = np.moveaxis(mask_array, 0, -1)
    mask = np.zeros([mask_array.shape[0], mask_array.shape[1], 1], np.uint8)
    mask_areas = []
    for i in range(num_instances):
        curr_mask = mask_array[:, :, i:(i+1)].astype(np.uint8)
        mask += curr_mask * instance_colors[i]
        mask_areas.append(int(np.sum(curr_mask)))
    # store viz of instances
    visualizer = Visualizer(img[:, :, ::-1], metadata={}, scale=0.5)
    out = visualizer.draw_instance_predictions(predictions["instances"].to("cpu"))
    return mask, mask_areas, instance_colors, num_instances, out.get_image()[:, :, ::-1], pred_time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, help="path to class-agnostic model",
                        default='../models/ABB_FAT_dopose_SPS_v3.pth')
    parser.add_argument("--debug", action="store_true", help="debug on local machine without socket")
    parser.add_argument("--img-path", type=str, default="../datasets/test-image.color.png", help="path to img for debug purposes")
    parser.add_argument("--visualize", action="store_true", help="visualize instances")
    args = parser.parse_args()
    
    # load model
    args.model_path = os.path.abspath(args.model_path)
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
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.05
    predictor = DefaultPredictor(cfg)

    socket_address = "/home/vision/abb-vision/cache/apick_nn_inst_seg.unixsock"
    try:
        os.unlink(socket_address)
    except OSError:
        if os.path.exists(socket_address):
            os.remove(socket_address)

    if args.debug:
        img = cv2.imread(args.img_path)
        mask, mask_areas, instance_colors, num_instances, viz_img, pred_time = get_predictions(predictor, img) 
        print("areas: {}, colors: {}, count: {}".format(mask_areas, instance_colors, num_instances))
        cv2.imshow("", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.bind(socket_address)
        print(f"Listening")
        s.listen()
        while True:
            conn, addr = s.accept()
            with conn:
                print(f"Connected by {addr}")
                payload = conn.recv(4096)
                sock_time = time.time()
                if payload:
                    payload = payload.decode("utf-8").rstrip("\x00")
                    print("received from client: {}".format(payload))
                    payload = json.loads(payload)
                    data = []
                    for key in payload:
                        cam_id = payload[key]["id"]
                        # get appropriate rgb img
                        rgb_path = os.path.abspath(payload[key]["rgb"])
                        result_path = os.path.abspath(payload[key]["result"])
                                            
                        img = cv2.imread(rgb_path)

                        # # crop region
                        if payload[key]["camera_roi"]["enable"]:
                            # Get region of interest
                            width = payload[key]["camera_roi"]["roi_width"]
                            height = payload[key]["camera_roi"]["roi_height"]
                            if width > 20 and height > 20:
                                roi_start = [payload[key]["camera_roi"]["start_x"], payload[key]["camera_roi"]["start_y"]]
                                roi_end = [roi_start[0]+width, roi_start[1]+height]

                                # draw rectangle using roi and crop everything else
                                roi_mask = np.zeros(img.shape[:2], dtype="uint8")
                                roi_mask = cv2.rectangle(roi_mask, roi_start, roi_end, (255,255,255),-1)
                                img = cv2.bitwise_and(img, img, mask=roi_mask)
                            else:
                                print("failed to crop image. Minimum height and width requirements not met.")

                        mask, mask_areas, instance_colors, num_instances, viz_img, pred_time = get_predictions(predictor, img) 
                        cv2.imwrite(result_path, mask)
                        result_path = result_path.replace(".png", ".visualization.png")
                        cv2.imwrite(result_path, viz_img)
                        sock_time  = time.time() - sock_time
                        print("socket time: {} pred time: {}".format(sock_time, pred_time))
                        data.append({"cam_id": cam_id, "segment_count": num_instances, "areas": mask_areas, "colors": instance_colors.tolist()})
                    response = { "status": "success", "data": data}
                    response = json.dumps(response).encode("utf-8")
                    conn.sendall(response)

                    if args.visualize:
                        cv2.imshow("", mask)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()



if __name__ == '__main__':
    main()
