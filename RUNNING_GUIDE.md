# How to Run the Brain Tumor Detection Project

This guide will walk you through setting up and running the application from scratch.

## Prerequisites
You must have **Python 3.8+** installed on your computer.

---

## Step 1: Install Requirements
Before running the application, you need to install the required Python libraries. 

1. Open your terminal or command prompt.
2. Navigate to the project directory (`Brain-Tumor-Detection-master/`).
3. Run the following command to install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```

*(Note: Depending on your Python installation, you might need to use `pip3` instead of `pip`).*

---

## Step 2: Run the Web Application
Once the dependencies are installed, you can start the Flask backend server.

1. In the same terminal window, run:
   ```bash
   python app.py
   ```
2. You should see output indicating that the Flask server is running (usually on `http://127.0.0.1:5000`).

---

## Step 3: Use the Web Interface
1. Open your preferred web browser (Chrome, Edge, Safari, Firefox).
2. Type the following into the address bar and hit Enter:
   **http://127.0.0.1:5000**
3. You will see the Brain Tumor Detector UI.
4. Open the `test_images/` folder on your computer.
5. Drag and drop any image from the `yes/` (tumor) or `no/` (no tumor) folders into the upload box on the website.
6. Click the **"Analyze Scan"** button.
7. The AI will process the image and immediately tell you if a tumor is detected, along with a confidence percentage.

---

## Optional: How to test the model in the terminal
If you want to quickly test the accuracy of the model on the 15 provided test images without using the web interface, you can run the test script:

```bash
python test_model.py
```
This will process all 15 images and print the pass/fail results directly in the terminal.

## Optional: How to retrain the model
If you wish to retrain the model from scratch (requires the full Kaggle dataset to be placed in a `dataset/` folder):

```bash
python train_model.py
```
This will build the MobileNetV2 architecture, run the data augmentation pipeline, train the model, and save a new `.h5` file in the `models/` folder.
