# Open Science AI/ML Showcase Examples
**AI models, tools, and deployment examples focused on marine ecosystem applications.**

**Contact:** [Michael.Akridge@noaa.gov](mailto:Michael.Akridge@noaa.gov)

<details>
  <summary><strong>Table of Contents</strong></summary>

1. [Overview](#overview)
2. [Models](#models)
   - [YOLO11 Architecture](#yolo11-architecture)
     - [Fish Object Detection](#fish-object-detection-model)
     - [Fish Segmentation](#fish-segmentation-model)
     - [Urchin Object Detection](#urchin-object-detection-models)
   - [YOLOv8 Architecture](#yolov8-architecture)
     - [Fish Object Detection](#fish-object-detection-yolov8)
3. [Desktop Tools & GUIs](#desktop-tools--guis)
   - [Simple GUI for Model Training](#simple-gui-model-training)
   - [Advanced GUI: Open Science AI Toolbox](#advanced-gui-open-science-ai-toolbox)
     - [YOLO11 Object Detection Training](#yolo11-object-detection-training)
     - [Dataset Preparation Tools](#dataset-prep)
     - [Dataset Viewer](#dataset-viewer)
     - [Advanced Metrics with Plotly](#advanced-metrics-using-plotly)
     - [Point Segmentation using SAM](#point-segmentation-using-sam-model)
     - [CoralNet Annotations to SAM Segmentation](#coralnet-to-sam-segmentation)
4. [Web Applications](#web-applications)
   - [Coral Analysis](#coral-analysis)
     - [Coral Feature Extraction](#coral-feature-extraction)
     - [Coral Segmentation & Class Mask Coverage](#coral-segmentation)
   - [Urchin Object Detection (Web)](#urchin-object-detection-web)
     - [Multiscale Patch-Based Detection](#multiscale-patch-detection)
     - [Batch Processing of Predictions](#batch-processing-of-urchin-detection)
     - [SfM Mosaic Detection](#sfm-mosaic-detection)
   - [Other Examples](#other-examples)
     - [Auto Segmentation with YOLO & SAM](#auto-segmentation)
     - [YOLO-World Inference](#yolo-world-inference)
     - [Fish vs. No Fish Examples](#fish-no-fish-examples)
   - [Depth Estimation Using ML](#depth-estimation)
5. [Cloud Deployments](#cloud-deployments)
   - [Google Cloud Shell Deployment](#google-cloud-shell-deployment)
   - [One-Click Cloud Deployment](#one-click-cloud-deployment)
   - [Batch Processing with Job Management](#batch-processing)
6. [Mobile Deployment](#mobile-deployment)
   - [Apple iOS Object Detector Deployment](#apple-ios-deployment)
7. [Disclaimer & License](#disclaimer--license)

</details>


## Models

### YOLO11 Architecture

#### Fish Object Detection Model
- **Model Card:** [Hugging Face](https://huggingface.co/akridge/yolo11-fish-detector-grayscale)
- **Example GIF:**
  ![](./visuals/yolo11n_ai_test.gif)

#### Fish Segmentation Model
- **Model Card:** [Hugging Face](https://huggingface.co/akridge/yolo11-segment-fish-grayscale)
- **Example GIF:**
  ![](./visuals/yolo11n-seg.gif)

#### Urchin Object Detection Models
- **Nano:** [Model Card](https://huggingface.co/akridge/yolo11n-sea-urchin-detector)
- **Medium:** [Model Card](https://huggingface.co/akridge/yolo11m-sea-urchin-detector)
- **XLarge:** [Model Card](https://huggingface.co/akridge/yolo11x-sea-urchin-detector)
  
  ![](./visuals/urchins/urchin_xl.jpg)

### YOLOv8 Architecture

#### Fish Object Detection
- **Model Card:** [Hugging Face](https://huggingface.co/akridge/yolo8-fish-detector-grayscale)

---

## Desktop Tools & GUIs

### Simple GUI for Model Training
*(YOLO11 (Ultralytics) Wrapper using PyQT5)*
- **Screenshot:**
  ![](./visuals/open_science_ai_gui/gui_01.png)

### Advanced GUI: Open Science AI Toolbox
- **Overview Screenshot:**
  ![](./visuals/open_science_ai_gui/gui_03.png)

#### YOLO11 Object Detection Training
- **Screenshot:**
  ![](./visuals/open_science_ai_gui/gui_04.png)

#### Dataset Preparation Tools
- *Placeholder for dataset preparation tools.*

#### Dataset Viewer
- Upload a `dataset.yaml` file to view images and bounding box overlays for dataset QA/QC.
- **Screenshot:**
  ![](./visuals/open_science_ai_gui/gui_05.png)

#### Advanced Metrics using Plotly
- Upload a `Training results.csv` file to generate training metrics charts.
- **Screenshot:**
  ![](./visuals/open_science_ai_gui/gui_06.png)

#### Point Segmentation using SAM Model
- **Screenshot:**
  ![](./visuals/open_science_ai_gui/gui_07.png)

#### CoralNet Annotations to SAM Segmentation
- Example CoralNet dataset (CSV format):
```
  csv | # example dataset
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
- **Screenshot:**
  ![](./visuals/open_science_ai_gui/gui_08.png)
## Web Applications

### Coral Analysis

#### Coral Feature Extraction
- EfficientNet-based feature extraction.
- **Screenshot:**
  ![](./visuals/streamlit_examples/streamlit_01_laptop_cpu_coralnet_classifer.png)

#### Coral Segmentation & Class Mask Coverage
- **Coral Instance Segmentation (YOLO11):**
  ![](./visuals/coral_seg_01.png)
- **Coverage Visualization:**
  ![](./visuals/coral_seg_02.png)

### Urchin Object Detection (Web)
  
#### Multiscale Patch-Based Detection
- **Screenshots:**
  ![](./visuals/patch_based_yolo_detection.png)
  ![](./visuals/20241021_urchin_model_performance_patch_comparision.png)

#### Batch Processing of Predictions
- **GIF Example:**
  ![](./visuals/urchin_batch_processing.gif)

#### SfM Mosaic Detection
- Using a crop-based approach.
- **GIF Example:**
  ![](./visuals/sfm_examples/SFM_01_mosaic_object_detection.gif)

### Other Web Applications

#### Auto Segmentation with YOLO & SAM
- **Screenshot:**
  ![](./visuals/auto_sam_01.png)

#### YOLO-World Model Inference
- Pre-trained “prompt-then-detect” model with prompt “fish”.
- **Screenshot:**
  ![](./visuals/yoloworld_01.png)

#### Depth Estimation Using Machine Learning
- **Project Repo:** [ml-depth-pro on GitHub](https://github.com/MichaelAkridge-NOAA/ml-depth-pro)
- **Depth Estimation Visualization App:** [ML Depth Pro](https://connect.fisheries.noaa.gov/ml_depth_pro/)
  
  <a href="https://connect.fisheries.noaa.gov/ml_depth_pro/" target="_blank">
    <img src="./visuals/ml-depth/ml_depth_01.png" alt="Depth Estimation Viewer">
  </a>

- **YOLO + Apple Depth Pro Integration:**  
  Combines YOLO11 object detection with depth estimation (including focal length prediction & scaling based on ground truth).  
  **Before/After Comparison:**
  ![](./visuals/ml-depth/ml_depth_02.png)
- **Animated Depth Visualization:**
  ![](./visuals/ml-depth/ml_depth_03.gif)

#### Fish vs. No Fish Examples
- **Screenshots:**
  ![](./visuals/fish_no_fish/01.png)
  ![](./visuals/fish_no_fish/03.png)

---

## Cloud Deployments

### Google Cloud Shell Deployment
- **Overview:**  
  Custom containerized (Docker) deployment of a Python-based Streamlit web app that runs object detection on data from a Google Cloud Bucket.
- **GIF Example:**
  ![](./visuals/google_cloud_shell/processing.gif)

### One-Click Cloud Deployment
- **Deploy with a Single Click:**
  [![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FMichaelAkridge-NOAA%2FFish-or-No-Fish-Detector&cloudshell_git_branch=MOUSS_2016&cloudshell_print=cloud-shell-readme.txt&cloudshell_workspace=google-cloud-shell&cloudshell_tutorial=TUTORIAL.md)

### Batch Processing with Job Management
- Enables a job queue system for starting and stopping object detection processing.
- **Screenshot:**
  ![](./visuals/batch_job_system_nodd.png)

---

## Mobile Deployment

### Apple iOS Object Detector Deployment
- **Overview:**  
  YOLO11 fish detection model converted to CoreML and deployed using an Ultralytics app template.
- **Model Link:**  
  [mouss-fish-Core-ML model on Hugging Face](https://huggingface.co/akridge/yolo11-fish-detector-grayscale/commit/6bc514313f7a25df726bd0d20a2b3d8787d53476)
- **Performance:**  
  Achieves >60 FPS on iPhone 15 Pro Max.
- **Diagram:**
  ![](./visuals/coreml_flow_diagram.png)

---

## Disclaimer & License

#### Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration or the United States Department of Commerce. All NOAA GitHub project content is provided on an “as is” basis, and the user assumes responsibility for its use. Any claims arising from the use of this project will be governed by applicable Federal law. References to specific commercial products, processes, or services do not imply endorsement by the Department of Commerce or any of its bureaus. The Department of Commerce seal and logo (or that of a DOC bureau) may not be used to imply endorsement of any commercial product or activity.

#### License
See [LICENSE.md](./LICENSE.md) for details.

## Web Applications

### Coral Analysis

#### Coral Feature Extraction
- EfficientNet-based feature extraction.
- **Screenshot:**
  ![](./visuals/streamlit_examples/streamlit_01_laptop_cpu_coralnet_classifer.png)

#### Coral Segmentation & Class Mask Coverage
- **Coral Instance Segmentation (YOLO11):**
  ![](./visuals/coral_seg_01.png)
- **Coverage Visualization:**
  ![](./visuals/coral_seg_02.png)

### Urchin Object Detection (Web)
  
#### Multiscale Patch-Based Detection
- **Screenshots:**
  ![](./visuals/patch_based_yolo_detection.png)
  ![](./visuals/20241021_urchin_model_performance_patch_comparision.png)

#### Batch Processing of Predictions
- **GIF Example:**
  ![](./visuals/urchin_batch_processing.gif)

#### SfM Mosaic Detection
- Using a crop-based approach.
- **GIF Example:**
  ![](./visuals/sfm_examples/SFM_01_mosaic_object_detection.gif)

### Other Examples

#### Auto Segmentation with YOLO & SAM
- **Screenshot:**
  ![](./visuals/auto_sam_01.png)

#### YOLO-World Model Inference
- Pre-trained “prompt-then-detect” model with prompt “fish”.
- **Screenshot:**
  ![](./visuals/yoloworld_01.png)

#### Depth Estimation Using Machine Learning
- **Project Repo:** [ml-depth-pro on GitHub](https://github.com/MichaelAkridge-NOAA/ml-depth-pro)
- **Depth Estimation Visualization App:** [ML Depth Pro](https://connect.fisheries.noaa.gov/ml_depth_pro/)
  
  <a href="https://connect.fisheries.noaa.gov/ml_depth_pro/" target="_blank">
    <img src="./visuals/ml-depth/ml_depth_01.png" alt="Depth Estimation Viewer">
  </a>

- **YOLO + Apple Depth Pro Integration:**  
  Combines YOLO11 object detection with depth estimation (including focal length prediction & scaling based on ground truth).  
  **Before/After Comparison:**
  ![](./visuals/ml-depth/ml_depth_02.png)
- **Animated Depth Visualization:**
  ![](./visuals/ml-depth/ml_depth_03.gif)

#### Fish vs. No Fish Examples
- **Screenshots:**
  ![](./visuals/fish_no_fish/01.png)
  ![](./visuals/fish_no_fish/03.png)

---

## Cloud Deployments

### Google Cloud Shell Deployment
- **Overview:**  
  Custom containerized (Docker) deployment of a Python-based Streamlit web app that runs object detection on data from a Google Cloud Bucket.
- **GIF Example:**
  ![](./visuals/google_cloud_shell/processing.gif)

### One-Click Cloud Deployment
- **Deploy with a Single Click:**
  [![Open in Cloud Shell](https://gstatic.com/cloudssh/images/open-btn.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https%3A%2F%2Fgithub.com%2FMichaelAkridge-NOAA%2FFish-or-No-Fish-Detector&cloudshell_git_branch=MOUSS_2016&cloudshell_print=cloud-shell-readme.txt&cloudshell_workspace=google-cloud-shell&cloudshell_tutorial=TUTORIAL.md)

### Batch Processing with Job Management
- Enables a job queue system for starting and stopping object detection processing.
- **Screenshot:**
  ![](./visuals/batch_job_system_nodd.png)

---

## Mobile Deployment

### Apple iOS Object Detector Deployment
- **Overview:**  
  YOLO11 fish detection model converted to CoreML and deployed using an Ultralytics app template.
- **Model Link:**  
  [mouss-fish-Core-ML model on Hugging Face](https://huggingface.co/akridge/yolo11-fish-detector-grayscale/commit/6bc514313f7a25df726bd0d20a2b3d8787d53476)
- **Performance:**  
  Achieves >60 FPS on iPhone 15 Pro Max.
- **Diagram:**
  ![](./visuals/coreml_flow_diagram.png)

---

## Disclaimer & License
#### Disclaimer
This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project content is provided on an ‘as is’ basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.

#### License
See [LICENSE.md](./LICENSE.md) file for details.