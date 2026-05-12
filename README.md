# Brain Tumor Detection Web Application

## Overview
This project is an end-to-end AI-powered web application designed to detect brain tumors from MRI scans. It utilizes a Deep Learning model built with Convolutional Neural Networks (CNNs) via Transfer Learning (MobileNetV2) to classify MRI images as either "Tumor Detected" or "No Tumor Detected." 

The project includes a fully functional, premium web interface that allows users to upload MRI scans, processes them in real-time on the backend, and displays the prediction along with a confidence score.

## Dataset
The model was trained using the publicly available **Brain MRI Images for Brain Tumor Detection** dataset from Kaggle. 
- The dataset consists of MRI scans categorized into two classes: `yes` (tumor present) and `no` (tumor absent).
- During training, the data was heavily augmented (using rotation, shifting, zooming, and flipping) to artificially increase the dataset size and improve the model's ability to generalize to new, unseen MRI scans.

## Model Architecture & Pipeline
1. **Preprocessing:** 
   - Before feeding the image to the model, an automated OpenCV script runs. It converts the image to grayscale, applies Gaussian blur, and uses contour detection to automatically crop the black, empty space around the brain.
   - The cropped brain image is then resized to `224x224` pixels and normalized.
2. **Transfer Learning (MobileNetV2):** 
   - The core architecture uses MobileNetV2 pre-trained on ImageNet.
   - The base layers act as robust feature extractors.
   - A custom classification head (GlobalAveragePooling -> Dense 128 -> Dropout 0.5 -> Dense 1 Sigmoid) was trained specifically on the Brain MRI dataset.
3. **Performance:** 
   - The trained model achieves over **90% validation accuracy** and near-perfect accuracy on the hold-out test images.

## Features
- **Custom Trained Model:** Python scripts are provided (`train_model.py`) that demonstrate exactly how the data was augmented and the model was built and fine-tuned.
- **Automated Preprocessing:** Intelligent cropping algorithm that focuses the AI entirely on the brain mass.
- **Interactive UI:** A glassmorphism-styled frontend built with HTML, CSS, and vanilla JavaScript. Includes drag-and-drop file uploading and animated confidence bars.
- **RESTful Backend:** Flask API that handles file uploads securely, runs the prediction pipeline, and returns JSON results.

## Project Structure
- `app.py`: The main Flask server backend.
- `train_model.py`: The script used to train the MobileNetV2 model.
- `test_model.py`: A terminal script to test model accuracy.
- `models/`: Contains the trained `new_custom_model.h5` file.
- `templates/` & `static/`: HTML and CSS for the web interface.
- `test_images/`: A small subset of 15 images used for live demonstration.
