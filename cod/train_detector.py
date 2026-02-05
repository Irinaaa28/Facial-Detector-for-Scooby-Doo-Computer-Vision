from sklearn.svm import LinearSVC
import cv2 as cv
from detect import *
from features import *
from dataset import *

def train_face_detector(descriptors, labels):
    clf = LinearSVC(C=0.01, class_weight="balanced", max_iter=10000)
    clf.fit(descriptors, labels)
    return clf

def get_hard_negatives(grouped_annotations, detector, window_size, hog_params, threshold = 0.1, max_per_image = 3, iou_neg=0.1):
    hard_negatives = []
    for image_path, anns in grouped_annotations.items():
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        gt_boxes = [ann["bbox"] for ann in anns]
        detections, scores = detect_faces_in_image(img, detector, window_size, hog_params, threshold)
        collected = 0
        for det, score in zip(detections, scores):
            ious = [intersection_over_union(det, gt) for gt in gt_boxes]
            if max(ious) < iou_neg:
                patch = img[det[1]:det[3], det[0]:det[2]]
                if patch.size == 0:
                    continue
                patch = cv.resize(patch, window_size)
                hard_negatives.append(patch)
                collected += 1
            if collected >= max_per_image:
                break
    return hard_negatives