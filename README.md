# Open Science AI/ML Showcase Examples
Suite of AI models, tools, and deployment examples focused on marine ecosystem applications.
---
## Table of Contents

- [Models](#models)
  - [YOLO11 Architecture](#yolo11-architecture)
    - [Fish Object Detection Model](#fish-object-detection-model)
    - [Fish Segmentation Model](#fish-segmentation-model)
    - [Urchin Object Detection Models](#urchin-object-detection-models)
  - [YOLOv8 Architecture](#yolov8-architecture)
    - [Fish Object Detection](#fish-object-detection)
- [Open Science AI Toolbox | Desktop GUI](#open-science-ai-toolbox-desktop-gui)
  - [Simple GUI / Model Training](#simple-gui--model-training)
  - [Advanced GUI / Open Science AI Toolbox](#advanced-gui--open-science-ai-toolbox)
    - [YOLO11 (Ultralytics) Object Detection Training](#yolo11-ultralytics-object-detection-training)
    - [Dataset Prep](#dataset-prep)
    - [Dataset Viewer](#dataset-viewer)
    - [Advanced Metrics using Plotly](#advanced-metrics-using-plotly)
    - [Point Segmentation using SAM Model](#point-segmentation-using-sam-model)
    - [CoralNet Point Annotations to Segmentation Layers using SAM](#coralnet-point-annotations-to-segmentation-layers-using-sam)
- [AI/ML Web App Examples](#web-app-examples)
  - [Coral Feature Extraction](#coral-feature-extraction)
  - [Urchin Object Detection on SfM Mosaics](#urchin-object-detection-on-sfm-mosaics)
  - [Fish Detection Examples](#fish-no-fish-examples)
- [Cloud Deployment of Object Detection Models](#cloud-deployment-of-object-detection-models)
  - [Google Cloud Shell Deployment](#google-cloud-shell-deployment)
  - [One Click Startup Deployment](#one-click-startup-deployment)
- [Mobile Deployment](#mobile-deployment)
  - [Apple iOS Deployment of Object Detectors](#apple-ios-deployment-of-object-detectors)

---

## Models
### YOLO11 Architecture
#### Fish Object Detection Model
- [Model Card](https://huggingface.co/akridge/yolo11-fish-detector-grayscale)
![](./visuals/yolo11n_ai_test.gif)
#### Fish Segmentation Model
- [Model Card](https://huggingface.co/akridge/yolo11-segment-fish-grayscale)
![](./visuals/yolo11n-seg.gif)

#### Urchin Object Detection Models
- **Nano:** [Model Card](https://huggingface.co/akridge/yolo11n-sea-urchin-detector)
- **Medium:** [Model Card](https://huggingface.co/akridge/yolo11m-sea-urchin-detector)
- **XLarge:** [Model Card](https://huggingface.co/akridge/yolo11x-sea-urchin-detector)

![](./visuals/urchins/urchin_xl.jpg)

### YOLOv8 Architecture
#### Fish Object Detection
- [Model Card](https://huggingface.co/akridge/yolo8-fish-detector-grayscale)
---

## Open Science AI Toolbox Desktop GUI
A desktop GUI for performing repeatable AI tasks with ease.
### Simple GUI / Model Training
*(YOLO11 (Ultralytics) Wrapper using PyQT5)*
![Simple GUI](./visuals/open_science_ai_gui/gui_01.png)

### Advanced GUI / Open Science AI Toolbox

![Advanced GUI](./visuals/open_science_ai_gui/gui_03.png)

#### YOLO11 (Ultralytics) Object Detection Training
![Training GUI](./visuals/open_science_ai_gui/gui_04.png)

#### Dataset Prep
*Placeholder for dataset preparation tools.*

#### Dataset Viewer
Upload a `dataset.yaml` file to view images and bounding box overlays for dataset QA/QC.

![Dataset Viewer](./visuals/open_science_ai_gui/gui_05.png)

#### Advanced Metrics using Plotly
Upload a `Training results.csv` file to generate various charts of common training metrics.

![Metrics Chart](./visuals/open_science_ai_gui/gui_06.png)

#### Point Segmentation using SAM Model

![Point Segmentation](./visuals/open_science_ai_gui/gui_07.png)

#### CoralNet Point Annotations to Segmentation Layers using SAM

Below is an example of a CoralNet dataset:

```csv
# example dataset
name,label_code,row,column
"./OAH-2749_2016_B_04.JPG",TURFR,614,2903
"./OAH-2749_2016_B_04.JPG",TURFH,692,1763
"./OAH-2749_2016_B_04.JPG",TURFR,985,704
"./OAH-2749_2016_B_04.JPG",TURFH,1205,3443
"./OAH-2749_2016_B_04.JPG",TURFH,2387,100
"./OAH-2749_2016_B_04.JPG",MOEN,1814,3221
"./OAH-2749_2016_B_04.JPG",SAND,1886,1092
"./OAH-2749_2016_B_04.JPG",TURFH,1930,1802
"./OAH-2749_2016_B_04.JPG",TURFH,1933,2825
"./OAH-2749_2016_B_04.JPG",TURFR,1470,887

```
![app](./visuals/open_science_ai_gui/gui_08.png)
# Web App Examples
Using Streamlit python libary for easy web framework. 

## Coral Feature Extraction
- efficientnet feature extraction

![app](./visuals/streamlit_examples/streamlit_01_laptop_cpu_coralnet_classifer.png)

## Coral Segmentation
### Yolo11 instance segmentation of corals
![app](./visuals/coral_seg_01.png)
### Coral Class Mask Coverage % of Total Image Area
![app](./visuals/coral_seg_02.png)

## Urchin Object Detection - Batch Processing of Predictions

![app](./visuals/urchin_batch_processing.gif)

## Urchin Object Detection on SfM Mosaics
Using crop based approach. 

![app](./visuals/sfm_examples/SFM_01_mosaic_object_detection.gif)

# Auto Segmentation Datasets & Models using YOLO & Segment Anythign Model(SAM)

![app](./visuals/auto_sam_01.png)

# YOLO-World Model Inference Test Results
- Using pre-trained "prompt-then-detect" model
- Prompt class ‘fish’
- Use case: Starting model to help with labeling data for larger, common objects like fish, turtles, seals. Does not perform well for less common objects like coral and urchins.
![app](./visuals/yoloworld_01.png)

# Depth Estimation Using Machine Learning (Apples ML Depth Pro) on Underwater Marine Imagery
- Github: https://github.com/MichaelAkridge-NOAA/ml-depth-pro
- Depth Estimation Visualization Viewer App https://connect.fisheries.noaa.gov/ml_depth_pro/

<a href="https://connect.fisheries.noaa.gov/ml_depth_pro/" target="_blank">
  <img src="./visuals/ml-depth/ml_depth_01.png" alt="screenshot">
</a>

### YOLO + Apple Depth Pro 
#### With Focal Length Prediction & Scaling Adjustments based on Known Ground Truth Distances
The following uses a trained YOLO11 object detection model combined with Depth Pro to calculate the depth of objects in images and apply scaling adjustments based on known ground truth distances. Applying a scaling factor derived from ground truth bounding boxes to improve depth accuracy across the image. When ground truth is missing, a default bounding box is used to maintain consistency in depth estimation. Below is before and after.
![](./visuals/ml-depth/ml_depth_02.png)

### Animated Depth Visualization
![](./visuals/ml-depth/ml_depth_03.gif)

## Fish no Fish Examples 
## Web App Examples
![app](./visuals/fish_no_fish/01.png)
![app](./visuals/fish_no_fish/03.png)


# Cloud Deployment of Object Detection Model (w/ Google Cloud Shell)
More information: 
- Custom Containezed (Docker) Deployment 
- Python based web app (Streamlit) 
- Deploys & Runs Inference using object detection model in Google Cloud (shell) on Google Cloud Bucket data(NODD)
- <b>Results</b>: Generates Model Predections in Ai-Ready YOLO format to be used during training

![app](./visuals/google_cloud_shell/processing.gif)

## One Click Startup Deployment
[![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FMichaelAkridge-NOAA%2FFish-or-No-Fish-Detector&cloudshell_git_branch=MOUSS_2016&cloudshell_print=cloud-shell-readme.txt&cloudshell_workspace=google-cloud-shell&cloudshell_tutorial=TUTORIAL.md)

![app](./visuals/google_cloud_shell/gcs_02.png)


# Object Detection Batch Processing & Job Mangagmnet via DB 
placeholder


# Mobile Phone (Apple IOS) Deployment of Object Detectors
- YOLO11 fish detection model loaded and tested on IOS natively 
- Using CoreML converted model and ultralytics app template. Export of: 
[mouss-fish-Core-ML model](https://huggingface.co/akridge/yolo11-fish-detector-grayscale/commit/6bc514313f7a25df726bd0d20a2b3d8787d53476)
- Solid performance(>60FPS) on Iphone 15 pro max 