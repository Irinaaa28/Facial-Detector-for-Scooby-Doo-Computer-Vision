import os
from collections import defaultdict
import cv2 as cv
import numpy as np
import random

def load_annotations(train_dir):
    annotations = []
    annotation_files = [f for f in os.listdir(train_dir) if f.endswith('.txt')]

    for ann_file in annotation_files:
        ann_path = os.path.join(train_dir, ann_file)
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue
                img_name = parts[0]
                xmin = int(parts[1])
                ymin = int(parts[2])
                xmax = int(parts[3])
                ymax = int(parts[4])
                label = parts[5]

                folder_name = ann_file.replace('_annotations.txt', '')
                img_path = os.path.join(train_dir, folder_name, img_name)

                annotation = {
                    "image_path": img_path,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "label": label
                }
                annotations.append(annotation)

    return annotations

def group_annotations_by_image(annotations):
    grouped_annotations = defaultdict(list)
    for ann in annotations:
        grouped_annotations[ann["image_path"]].append(ann)
    return grouped_annotations

def intersection_over_union(bbox_a, bbox_b):
    x_a = max(bbox_a[0], bbox_b[0])
    y_a = max(bbox_a[1], bbox_b[1])
    x_b = min(bbox_a[2], bbox_b[2])
    y_b = min(bbox_a[3], bbox_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
    box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

    iou = inter_area / float(box_a_area + box_b_area - inter_area)
    return iou

def non_maximum_suppression(image_detections, image_scores, image_size):

    # xmin, ymin, xmax, ymax
    x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
    y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
    print(x_out_of_bounds, y_out_of_bounds)
    image_detections[x_out_of_bounds, 2] = image_size[1]
    image_detections[y_out_of_bounds, 3] = image_size[0]
    sorted_indices = np.flipud(np.argsort(image_scores))
    sorted_image_detections = image_detections[sorted_indices]
    sorted_scores = image_scores[sorted_indices]

    is_maximal = np.ones(len(image_detections)).astype(bool)
    iou_threshold = 0.3
    for i in range(len(sorted_image_detections) - 1):
        if is_maximal[i] == True: 
            for j in range(i + 1, len(sorted_image_detections)):
                if is_maximal[j] == True: 
                    if intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                    else: 
                        c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                        c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                        if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                            is_maximal[j] = False
    return sorted_image_detections[is_maximal], sorted_scores[is_maximal]

def extract_training_patches(grouped_annotations, window_sizes, negatives_per_image = 5):
    data = {ws: {"pos": [], "neg": []} for  ws in window_sizes}

    for image_path, anns in grouped_annotations.items():
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        H, W = img.shape
        face_boxes = [ann["bbox"] for ann in anns]

        # POZITIVE PATCHES
        for (xmin, ymin, xmax, ymax) in face_boxes:
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(W, xmax)
            ymax = min(H, ymax)
            w = xmax - xmin
            h = ymax - ymin
            if w <= 0 or h <= 0:
                continue
            ratio = w / h 
            window = (60, 80) if ratio < 0.85 else (64, 64)

            patch = img[ymin:ymax, xmin:xmax]

            patch = cv.resize(patch, window)
            data[window]["pos"].append(patch)

        # NEGATIVE PATCHES
        for window in window_sizes:
            pw, ph = window
            tries = 0
            collected = 0
            while collected < negatives_per_image and tries < negatives_per_image * 10:
                tries += 1
                x = random.randint(0, W - pw)
                y = random.randint(0, H - ph)
                candidate_box = (x, y, x + pw, y + ph)
                overlaps = [intersection_over_union(candidate_box, fb) for fb in face_boxes]
                if max(overlaps) < 0.1:
                    patch = img[y: y + ph, x: x + pw]
                    data[window]["neg"].append(patch)
                    collected += 1

    return data

def extract_patches_by_character(grouped_annotations, window_sizes, negatives_per_image=5):
    characters = ["fred", "daphne", "velma", "shaggy"]
    data = {char: {ws: {"pos": [], "neg": []} for ws in window_sizes} for char in characters}

    for image_path, anns in grouped_annotations.items():
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None: 
            continue
        H, W = img.shape

        # POSITIVES
        for ann in anns:
            label = ann["label"]
            if label not in characters:
                continue
            xmin, ymin, xmax, ymax = ann["bbox"]
            patch = img[max(0, ymin):min(H, ymax), max(0, xmin): min(W, xmax)]
            if patch.size == 0:
                continue
            w, h = xmax - xmin, ymax - ymin
            ratio = w / h if h > 0 else 1
            chosen_win = (60, 80) if ratio < 0.85 else (64, 64)
            patch_res = cv.resize(patch, chosen_win)
            data[label][chosen_win]["pos"].append(patch_res)

        # NEGATIVES
        face_boxes = [ann["bbox"] for ann in anns]
        for char in characters:
            for window in window_sizes:
                pw, ph = window
                collected = 0
                tries = 0
                while collected < negatives_per_image and tries < 50:
                    tries += 1
                    x, y = random.randint(0, W - pw), random.randint(0, H - ph)
                    candidate_box = (x, y, x + pw, y + ph)
                    overlaps = [intersection_over_union(candidate_box, fb) for fb in face_boxes]
                    if max(overlaps) < 0.1:
                        patch = img[y: y + ph, x: x + pw]
                        data[char][window]["neg"].append(patch)
                        collected += 1
    return data

