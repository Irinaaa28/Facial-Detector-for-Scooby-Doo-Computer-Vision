import os
import cv2 as cv
import shutil

def convert_to_yolo_format(img_width, img_height, bbox):
    x_min, y_min, x_max, y_max = bbox
    dw = 1. / img_width
    dh = 1. / img_height
    w = x_max - x_min
    h = y_max - y_min
    x_center = x_min + w / 2.0
    y_center = y_min + h / 2.0
    return (x_center * dw, y_center * dh, w * dw, h * dh)

def generate_yolo_labels(train_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    annotation_files = [f for f in os.listdir(train_dir) if f.endswith('_annotations.txt')]
    for ann_file in annotation_files:
        ann_path = os.path.join(train_dir, ann_file)
        char_name = ann_file.replace('_annotations.txt', '')
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                img_name = parts[0]
                bbox = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
                img_path = os.path.join(train_dir, char_name, img_name)
                img = cv.imread(img_path)
                if img is None:
                    continue
                h, w, _ = img.shape
                yolo_bbox = convert_to_yolo_format(w, h, bbox)
                class_id = 0
                label_file_name = f"{char_name}_{os.path.splitext(img_name)[0]}.txt"
                label_path = os.path.join(output_dir, label_file_name)
                with open(label_path, 'a') as lf:
                    lf.write(f"{class_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}\n")

def generate_yolo_labels_per_character(train_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    char_to_id = {"fred": 0, "daphne": 1, "shaggy": 2, "velma": 3}
    annotation_files = [f for f in os.listdir(train_dir) if f.endswith('_annotations.txt')]
    for ann_file in annotation_files:
        ann_path = os.path.join(train_dir, ann_file)
        folder_name = ann_file.replace('_annotations.txt', '')
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                img_name = parts[0]
                bbox = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
                char_label = parts[5].lower()
                if char_label not in char_to_id:
                    continue
                class_id = char_to_id[char_label]
                img_path = os.path.join(train_dir, folder_name, img_name)
                img = cv.imread(img_path)
                if img is None:
                    continue
                h, w, _ = img.shape
                yolo_bbox = convert_to_yolo_format(w, h, bbox)
                label_file_name = f"{folder_name}_{os.path.splitext(img_name)[0]}.txt"
                label_path = os.path.join(output_dir, label_file_name)
                with open(label_path, 'a') as lf:
                    lf.write(f"{class_id} {' '.join([f'{x:.6f}' for x in yolo_bbox])}\n")
    print("YOLO label generation completed successfully!")

def organize_images_for_yolo(base_train_dir, yolo_train_dir):
    if not os.path.exists(yolo_train_dir):
        os.makedirs(yolo_train_dir)
    characters = ['fred', 'daphne', 'velma', 'shaggy']
    for char in characters:
        char_dir = os.path.join(base_train_dir, char)
        if not os.path.exists(char_dir):
            print(f"Directory {char_dir} not found")
            continue
        print(f"Processing folder {char_dir}...")
        images = [f for f in os.listdir(char_dir) if f.endswith('.jpg')]
        for img_name in images:
            src_path = os.path.join(char_dir, img_name)
            new_img_name = f"{char}_{img_name}"
            dst_path = os.path.join(yolo_train_dir, new_img_name)
            shutil.copy2(src_path, dst_path)

    print("Image organization for YOLO completed successfully!")

def organize_test_images(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if not os.path.exists(src_dir):
        print(f"Source directory{src_dir} not found")
        return
    images = [f for f in os.listdir(src_dir) if f.endswith('.jpg')]
    print(f"Processing test images from {src_dir} ...")
    for img_name in images:
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        shutil.copy2(src_path, dst_path)
    print("Test image organization completed successfully!")
    
                    
organize_images_for_yolo("antrenare", "yolo/images/train")
organize_test_images("validare/validare", "yolo/images/val")
generate_yolo_labels("antrenare", "yolo/task1/labels/train")
generate_yolo_labels_per_character("antrenare", "yolo/task2/labels/train")
