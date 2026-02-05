from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data='yolo/task1/data.yaml', epochs=50, imgsz=640, device='cpu') # Too slow
