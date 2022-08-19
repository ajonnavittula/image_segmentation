#!/usr/bin/env python
import os
import cv2
import numpy as np

# Saves every instance mask individually corresponding to post-processing requirements
def rearrange():

    # get absolute path
    image_path = "../datasets/sps/rgb/"
    image_ids = next(os.walk([image_path][0]))[2]
    image_ids.sort()
    depth_path = "../datasets/sps/depth/"
    depth_ids = next(os.walk([depth_path][0]))[2]
    depth_ids.sort()
    msk_path = "../datasets/sps/masks/"
    msk_ids = next(os.walk([msk_path][0]))[2]
    msk_ids.sort()

    for i in range(len(image_ids)):
        rgb_p = "../postprocessing/" + image_ids[i][:-4] + "/rgb/"
        depth_p = "../postprocessing/" + image_ids[i][:-4] + "/depth/"
        masks_p = "../postprocessing/" + image_ids[i][:-4] + "/masks/"
        cloud_p = "../postprocessing/" + image_ids[i][:-4] + "/cloud/"

        try:
            os.makedirs(rgb_p)
            os.makedirs(depth_p)
            os.makedirs(masks_p)
            os.makedirs(cloud_p)
        except OSError as error:
            print(error)

        # Saving depth file in correct format - comment if already depth image
        a = np.load(depth_path + depth_ids[i])
        out = a*10000
        cv2.imwrite(depth_p + depth_ids[i][:-4] + ".png", out.astype(np.uint16))

        # Saving colored images to corresponding directory
        b = cv2.imread(image_path + image_ids[i])
        cv2.imwrite(rgb_p + image_ids[i], b)

    for j in range(len(msk_ids)):
        # Parsing out individual masks
        c = cv2.imread(msk_path + msk_ids[j])
        cv2.imwrite("../postprocessing/" + msk_ids[j][:-9] + "/masks/" + msk_ids[j][-8:], c)

if __name__ == '__main__':
    rearrange()
