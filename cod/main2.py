import os
import pickle
import numpy as np
from dataset import *
from features import *
from train_detector import *
from multiscale import *
from sklearn.svm import LinearSVC

# define HOG parameters and window sizes
hog_params = {
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "orientations": 9,
    "block_norm": "L2-Hys"
}
window_sizes = {(64, 64), (60, 80)}
characters = ["fred", "daphne", "velma", "shaggy"]
threshold = 0.5

train_dir = "antrenare"
annotations = load_annotations(train_dir)
grouped = group_annotations_by_image(annotations)
train_data = extract_patches_by_character(grouped, window_sizes, negatives_per_image=5)

if not os.path.exists("models_task2"):
    os.makedirs("models_task2")

for char in characters:
    print(f"Training detectors for character: {char} ...")
    for window in window_sizes:
        pos_patches = train_data[char][window]["pos"]
        neg_patches = train_data[char][window]["neg"]
        for other_char in characters:
            if other_char != char:
                neg_patches.extend(train_data[other_char][window]["pos"])
        if len(pos_patches) == 0:
            continue
        pos_descriptors = extract_hog_features(pos_patches, hog_params)
        neg_descriptors = extract_hog_features(neg_patches, hog_params)
        descriptors = np.vstack((pos_descriptors, neg_descriptors))
        labels = np.hstack((np.ones(len(pos_descriptors)),
                            np.zeros(len(neg_descriptors))))
        clf = train_face_detector(descriptors, labels)
        model_filename = f"models_task2/{char}_detector_{window[0]}x{window[1]}.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(clf, f)
    print(f"Detectors for character {char} trained and saved successfully.")

detectors = {}
for char in characters:
    detectors[char] = {}
    for ws in window_sizes:
        model_path = f"models_task2/{char}_detector_{ws[0]}x{ws[1]}.pkl"
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                detectors[char][ws] = pickle.load(f)

# validation_dir = "validare/validare"
validation_dir = "testare"
image_files = [f for f in os.listdir(validation_dir) if f.endswith('.jpg')]
all_detections = []
all_scores = []
all_file_names = []
all_labels = []

for img_name in image_files:
    print(f"Processing image: {img_name} ...")
    img_path = os.path.join(validation_dir, img_name)
    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        continue
    for char in characters:
        char_detections = []
        char_scores = []
        for scaled_img, scale in pyramid(img):
            for ws in window_sizes:
                if ws not in detectors[char]:
                    continue
                detections, scores = detect_faces_in_image(scaled_img, detectors[char][ws], ws, hog_params, threshold)
                if len(detections) > 0:
                    detections = (detections / scale).astype(int)
                    char_detections.extend(detections)
                    char_scores.extend(scores)
        if len(char_detections) > 0:
            final_detections, final_scores = non_maximum_suppression(np.array(char_detections), np.array(char_scores), img.shape)
            for d, s in zip(final_detections, final_scores):
                all_detections.append(d)
                all_scores.append(s)
                all_file_names.append(img_name)
                all_labels.append(char)

results_per_char = {char: {"dets": [], "scores": [], "file_names": []} for char in characters}
for det, score, img_name, label in zip(all_detections, all_scores, all_file_names, all_labels):
    results_per_char[label]["dets"].append(det)
    results_per_char[label]["scores"].append(score)
    results_per_char[label]["file_names"].append(img_name)

output_dir = "333_Coman_IrinaElena/task2/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for char in characters:
    char_dets = np.array(results_per_char[char]["dets"])
    char_scores = np.array(results_per_char[char]["scores"])
    char_file_names = np.array(results_per_char[char]["file_names"])

    np.save(os.path.join(output_dir, f"detections_{char}.npy"), char_dets)
    np.save(os.path.join(output_dir, f"scores_{char}.npy"), char_scores)
    np.save(os.path.join(output_dir, f"file_names_{char}.npy"), char_file_names)

    print(f"Detections for character {char} saved successfully.")