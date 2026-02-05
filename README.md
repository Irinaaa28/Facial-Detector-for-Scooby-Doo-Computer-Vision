# Facial detector and recognition program for cartoon characters, Scooby-Doo edition

## Config
Libraries used in the implementation of this project and their versions:
* opencv-python==4.12.0.88
* numpy==2.1.2
* scikit-image==0.25.2
* scikit-learn==1.7.2  
* ultralytics==8.3.249

## About 
This project implements a complete computer vision pipeline for detecting and identifying the human characters from the **Scooby-Doo** series. The solution covers face localization (Task 1) for all human characters and individual character recognition (Task 2) for the main characters only: Fred, Daphne, Shaggy and Velma. Moreover, the solution uses two approaches, namely classical machine learning and deep learning.

## Features
#### Task 1: Face Detection
* Classical Approach: **HOG (Histogram of Oriented Gradients)** descriptors with a **Linear SVM** classifier
* Multi-scale detection using an Image Pyramid and Sliding Window
* Post-processing using **Non-Maximum Suppresion (NMS)**
* Achieved performance: 65.1% Average Precision  

#### Task 2: Character Recognition
* Classification of localized faces into 4 categories: Fred, Daphne, Shaggy and Velma
* One-vs-Rest classification using SVMs
* Achieved performace: 40.92% mean Average Precision 
    * 37.8% AP for Daphne
    * 27.7% AP for Fred
    * 26.5% AP for Shaggy
    * 71.7% AP for Velma  

#### Bonus: Deep Learning Integration for both tasks
* Implementation of a **YOLO (You Only Look Once)** detector
* Fine-tuned on a dataset of 4000 images 
* Achieved near-perfect results: 99.6% AP for Task 1 and 98.6% mAP for Task 2:
    * 97.9% AP for Daphne
    * 100% AP for Fred
    * 98.9% AP for Shaggy
    * 97.7% for Velma 

## How to run 

**Warning!** All the files and the commands below should be run in the root of the project, otherwise the project will fail due to path conflicts.

#### Install dependencies (if you don't have them)
``` 
pip install opencv-python scikit-image scikit-learn numpy ultralytics
``` 

#### Run Task 1
Run main.py file:
```
python -u cod/main.py
```
This file takes almost 7 minutes to run and generate solutions.  
The output path for task 1 is 333_Coman_IrinaElena/task1.

#### Run Task 2
Run main2.py file:
```
python -u cod/main2.py
```
This file takes almost 25 minutes to run and generate solutions.  
The output path for task 2 is 333_Coman_IrinaElena/task2.

#### Bonus
To achieve maximum performance and robustness, a secondary solution was implemented using the **YOLOv8 (You Only Look Once)** architecture. This deep learning model was specifically fine-tuned for the Scooby-Doo dataset.  

Due to the high computational cost of training neural networks, the model was trained using **Google Colab's Tesla T4 GPU**.  

The following steps are available for both tasks.

Run yolo.py to
* make a copy and rename every photo from the training directory 
* make a copy for every photo from the validating/testing directory
* generate labels in YOLO format 

```
python -u yolo/yolo.py
``` 

This project already has a model for each task, in yolo/runs/detect/train/weights for task 1 and yolo/runs/detect/train2/weights for task 2.  

Finally, save the results by running save_results.py
```
python -u yolo/save_results.py
```  

The output paths are 333_Coman_IrinaElena/bonus/task1 for Task 1 and 333_Coman_IrinaElena/bonus/task2 for Task 2.

#### If you want to create your own models, follow the next steps:  

Copy the images directory, which was generated earlier, from yolo directory and paste it to both yolo/task1 and yolo/task2 directories.  

Go to Google Colab, create a new notebook and set the hardware accelerator to **T4 GPU** (Edit -> Notebook settings).

Compress yolo/task1 or yolo/task2 directory to ZIP file, depending on which task you want to train the model for. 

Install ultralytics (if you don't have)  
```
!pip install ultralytics
```

Write in notebook the following code:
```
from google.colab import files
uploaded = files.upload()
```

Select the zipped file.
```
!unzip -q name_of_the_zipped_file.zip -d ./
``` 

Go to data.yaml, change the path to "/content", save it and close.

Then write and run this code:
```
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='/content/data.yaml',
    epochs=50,
    imgsz=640,
    device=0
)
```

It takes around 50-60 minutes to train the model.     

After the training finished, go to content/runs/detect/train/weights, download best.pt and move it to the local project, at yolo/runs/detect/train/weights for task 1 or yolo/runs/detect/train2/weights for task 2.   

Finally, save the results by running save_results.py
```
python -u yolo/save_results.py
```  

Finally, you have the results saved in the same format as the ones from classical machine learning approach.    

The output paths are 333_Coman_IrinaElena/bonus/task1 for Task 1 and 333_Coman_IrinaElena/bonus/task2 for Task 2.
