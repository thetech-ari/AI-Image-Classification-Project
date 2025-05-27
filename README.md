# AI Image Classification & Processing Project

This project explores the behavior of image classifiers using a pretrained MobileNetV2 model, Grad-CAM visualization, image occlusion experiments, and creative image filtering techniques.

## 📌 Overview

The project is divided into three parts:

### 🔹 Part 1: Basic Classifier + Grad-CAM
- Classifies an image (`dog1.jpg`) using a pretrained MobileNetV2 model.
- Uses Grad-CAM to visualize which areas of the image the model focuses on for prediction.
- Output: `heatmap_dog1.jpg`

### 🔹 Part 2: Occlusion Experiments
- Applies three occlusion methods (black box, blur, pixelation) to the most important image region.
- Measures how each occlusion affects classifier predictions.
- Output: `occluded_black_box.jpg`, `occluded_blur.jpg`, `occluded_pixel.jpg`

### 🔹 Part 3: Image Filters
- Applies various image filters including edge detection, sharpening, sepia tone, and a deep-fried artistic effect.
- Output: `filter_edge.jpg`, `filter_sharpened.jpg`, `filter_sepia.jpg`, `filter_deepfried.jpg`

## 🧠 Key Learnings

- Grad-CAM highlights which parts of an image are most influential for classification.
- Occlusion of key regions reduces classifier accuracy.
- Filters can alter visual features significantly, allowing experimentation with robustness and stylization.

## 🛠 Requirements

Install all dependencies with:
```
pip install -r requirements.txt
```

## 📁 Files

- `base_classifier.py` – runs the image classifier
- `base_classifier_gradcam.py` – adds Grad-CAM visualization
- `occlusion_classifier.py` – applies occlusions and reclassifies
- `image_filters.py` – applies image filters (including custom artistic ones)
- `dog1.jpg` – input image

## 📷 Image Outputs

All output images are saved in the same directory as the scripts. These include heatmaps, occluded images, and filtered versions.

---

Project inspired by coursework on AI model interpretability and image processing.
