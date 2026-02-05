# saves_results.py for Google Colab
import numpy as np
import os
import cv2 as cv
from ultralytics import YOLO

task = int(input("Task = "))

if task == 1:
    model = YOLO('/content/runs/detect/train/weights/best.pt')
    images_path = '/content/images/val'
    results = model(images_path, conf=0.25)

    all_bboxes = []
    all_scores = []
    all_filenames = []

    for r in results:
        img_name = os.path.basename(r.path)
        for box in r.boxes:
            coords = box.xyxy[0].cpu().numpy()
            score = box.conf[0].cpu().numpy()

            all_bboxes.append(coords)
            all_scores.append(score)
            all_filenames.append(img_name)

    output_path = '/content/333_Coman_IrinaElena/bonus/task1/'
    os.makedirs(output_path, exist_ok = True)
    np.save(os.path.join(output_path, "detections_all_faces.npy"), np.array(all_bboxes))
    np.save(os.path.join(output_path, "scores_all_faces.npy"), np.array(all_scores))
    np.save(os.path.join(output_path, "file_names_all_faces.npy"), np.array(all_filenames))
    print(".npy files saved successfully for task 1!")

elif task == 2:
    model = YOLO('/content/runs/detect/train2/weights/best.pt')
    images_path = '/content/images/val'
    results = model(images_path, conf=0.25)

    results_per_char = {0: "fred", 1: "daphne", 2: "shaggy", 3: "velma"}
    data_per_char = {char: {"bboxes": [], "scores": [], "filenames": []} for char in results_per_char.values()}

    for r in results:
        img_name = os.path.basename(r.path)
        for box in r.boxes:
            coords = box.xyxy[0].cpu().numpy()
            score = box.conf[0].cpu().numpy()
            class_id = int(box.cls[0]).cpu().numpy()
            if class_id in results_per_char:
                char_name = results_per_char[class_id]
                data_per_char[char_name]["bboxes"].append(coords)
                data_per_char[char_name]["scores"].append(score)
                data_per_char[char_name]["filenames"].append(img_name)

    output_path = '/content/333_Coman_IrinaElena/bonus/task2/'
    os.makedirs(output_path, exist_ok = True)
    for char_name, content in data_per_char.items():
        np.save(os.path.join(output_path, f"detections_{char_name}.npy"), np.array(content["bboxes"]))
        np.save(os.path.join(output_path, f"scores_{char_name}.npy"), np.array(content["scores"]))
        np.save(os.path.join(output_path, f"file_names_{char_name}.npy"), np.array(content["filenames"]))

else:
    print("invalid task number.")