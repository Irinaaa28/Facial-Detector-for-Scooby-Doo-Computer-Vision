from dataset import *
from features import *
from train_detector import *
from detect import *
from multiscale import *

import numpy as np
import os
import cv2 as cv

# define HOG parameters and window sizes
hog_params = {
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "orientations": 9,
    "block_norm": "L2-Hys"
}
window_sizes = {(64, 64), (60, 80)}

# load annotations from .txt files
train_dir = "antrenare"
annotations = load_annotations(train_dir)
grouped = group_annotations_by_image(annotations)

# extract training patches
train_data = extract_training_patches(grouped, window_sizes, negatives_per_image=5)
detectors = {}

for window in window_sizes:
    pos = train_data[window]["pos"]
    neg = train_data[window]["neg"]
    positive_descriptors = extract_hog_features(pos, hog_params)
    negative_descriptors = extract_hog_features(neg, hog_params)
    descriptors = np.vstack((positive_descriptors, negative_descriptors))
    labels = np.hstack((np.ones(len(positive_descriptors)), 
                        np.zeros(len(negative_descriptors))
                    ))
    clf = train_face_detector(descriptors, labels)
    # hard_negs = get_hard_negatives(grouped, clf, window, hog_params, threshold=0.4, max_per_image=3, iou_neg=0.1, use_pyramid=False)
    # all_neg = neg + hard_negs
    # negative_descriptors = extract_hog_features(all_neg, hog_params)
    # descriptors = np.vstack((positive_descriptors, negative_descriptors))
    # labels = np.hstack((np.ones(len(positive_descriptors)), 
    #                     np.zeros(len(negative_descriptors))
    #                 ))
    # clf = train_face_detector(descriptors, labels)
    detectors[window] = clf

# detections on validation images
# validation_dir = "validare/validare"
validation_dir = "testare"

all_detections = []
all_scores = []
all_file_names = []

image_files = [f for f in os.listdir(validation_dir) if f.endswith(".jpg")]
for img_name in image_files:
    print("Processing image:", img_name)
    img_path = os.path.join(validation_dir, img_name)
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Failed to load image: {img_path}")
        continue
    image_detections = []
    image_scores = []
    for scaled_img, scale in pyramid(img):
        for window in window_sizes:
            dets, scores = detect_faces_in_image(scaled_img, detectors[window], window, hog_params, threshold=0.3) # initial 0.5
            if len(dets) == 0:
                continue
            dets = (dets / scale).astype(int)
            image_detections.extend(dets)
            image_scores.extend(scores)
    print(f"{img_name}: {len(image_detections)} raw detections")
    if len(image_detections) > 0:
        final_detections, final_scores = non_maximum_suppression(np.array(image_detections), np.array(image_scores), img.shape)
        print(f"{img_name}: {len(final_detections)} final detections after NMS")
        for det, score in zip(final_detections, final_scores):
            all_detections.append(det)
            all_scores.append(score)
            all_file_names.append(img_name)

output_dir_task1 = "333_Coman_IrinaElena/task1/"
if not os.path.exists(output_dir_task1):
    os.makedirs(output_dir_task1)

np.save(os.path.join(output_dir_task1, "detections_all_faces.npy"), np.array(all_detections))
np.save(os.path.join(output_dir_task1, "scores_all_faces.npy"), np.array(all_scores))
np.save(os.path.join(output_dir_task1,"file_names_all_faces.npy"), np.array(all_file_names))

print("Final detections found:", len(all_detections))
