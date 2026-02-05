import numpy as np
import os
import cv2 as cv
from ultralytics import YOLO

path_to_model = 'yolo/runs/detect/train/weights/best.pt'
images_path = 'yolo/images/val'
output_path = '333_Coman_IrinaElena/bonus/task1/'

if not os.path.exists(path_to_model):
    print(f"Error: Model file not found at {path_to_model}")
    exit()

model = YOLO(path_to_model)

if not os.path.exists(output_path):
    os.makedirs(output_path)

print("Running inference on validation images...")
image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

all_bboxes = []
all_scores = []
all_filenames = []
results = model.predict(images_path, conf=0.1, device='cpu', stream=True)

for r in results:
    img_name = os.path.basename(r.path)
    for box in r.boxes:
        coords = box.xyxy[0].tolist()
        score = box.conf[0].item()

        all_bboxes.append(coords)
        all_scores.append(score)
        all_filenames.append(img_name)

np.save(os.path.join(output_path, "detections_all_faces.npy"), np.array(all_bboxes))
np.save(os.path.join(output_path, "scores_all_faces.npy"), np.array(all_scores))
np.save(os.path.join(output_path, "file_names_all_faces.npy"), np.array(all_filenames))
print(".npy files saved successfully for task 1!")
print(f"Final detections found: {len(all_bboxes)}")

path_to_model = 'yolo/runs/detect/train2/weights/best.pt'
images_path = 'yolo/images/val'
output_path = '333_Coman_IrinaElena/bonus/task2/'

if not os.path.exists(path_to_model):
    print(f"Error: Model file not found at {path_to_model}")
    exit()

model = YOLO(path_to_model)

if not os.path.exists(output_path):
    os.makedirs(output_path)

results_per_char = {0: "fred", 1: "daphne", 2: "shaggy", 3: "velma"}
data_per_char = {char: {"bboxes": [], "scores": [], "filenames": []} for char in results_per_char.values()}

results = model.predict(images_path, conf=0.1, device='cpu', stream=True)

for r in results:
    img_name = os.path.basename(r.path)
    for box in r.boxes:
        coords = box.xyxy[0].tolist()
        score = box.conf[0].item()
        class_id = int(box.cls[0].item())
        if class_id in results_per_char:
            char_name = results_per_char[class_id]
            data_per_char[char_name]["bboxes"].append(coords)
            data_per_char[char_name]["scores"].append(score)
            data_per_char[char_name]["filenames"].append(img_name)

for char_name, content in data_per_char.items():
    np.save(os.path.join(output_path, f"detections_{char_name}.npy"), np.array(content["bboxes"]))
    np.save(os.path.join(output_path, f"scores_{char_name}.npy"), np.array(content["scores"]))
    np.save(os.path.join(output_path, f"file_names_{char_name}.npy"), np.array(content["filenames"]))
print(".npy files saved successfully for task 2!")
print(f"Final detections found: {sum(len(content['bboxes']) for content in data_per_char.values())}")