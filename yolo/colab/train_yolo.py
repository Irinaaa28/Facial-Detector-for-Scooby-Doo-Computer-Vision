# YOLOv8 training script for Google Colab
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

task = int(input("Task = "))
if task == 1:
    results = model.train(data='/content/data.yaml', epochs=50, imgsz=640, device=0)
elif task == 2:
    results = model.train(data='/content/data2.yaml', epochs=50, imgsz=640, device=0)
else:
    print("Invalid task number.")