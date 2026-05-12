"""
Test script for the new custom model.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import cv2
import imutils
import numpy as np
import tf_keras as keras

MODEL_H5 = 'models/new_custom_model.h5'
IMG_WIDTH, IMG_HEIGHT = 224, 224

# If the model doesn't exist yet, warn the user
if not os.path.exists(MODEL_H5):
    print("WARNING: Model not found. Please run 'python train_model.py' first.")
    exit(1)

best_model = keras.models.load_model(MODEL_H5)

def crop_brain_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    return new_image

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = crop_brain_contour(image)
    image = cv2.resize(image, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    image = image / 255.
    image = np.expand_dims(image, axis=0)
    return image

def predict_single(image_path):
    img = preprocess_image(image_path)
    prob = float(best_model.predict(img, verbose=0)[0][0])
    if prob >= 0.5:
        return "TUMOR", round(prob * 100, 2)
    else:
        return "NO TUMOR", round((1 - prob) * 100, 2)

print("=" * 60)
print("TESTING TUMOR IMAGES (expected: TUMOR)")
print("=" * 60)
yes_dir = "test_images/yes"
tumor_correct = 0
tumor_total = 0
for f in sorted(os.listdir(yes_dir)):
    path = os.path.join(yes_dir, f)
    label, conf = predict_single(path)
    status = "PASS" if label == "TUMOR" else "FAIL"
    if label == "TUMOR": tumor_correct += 1
    tumor_total += 1
    print(f"  [{status}] {f:20s} -> {label:10s} ({conf}%)")

print("\n" + "=" * 60)
print("TESTING NON-TUMOR IMAGES (expected: NO TUMOR)")
print("=" * 60)
no_dir = "test_images/no"
no_correct = 0
no_total = 0
for f in sorted(os.listdir(no_dir)):
    path = os.path.join(no_dir, f)
    label, conf = predict_single(path)
    status = "PASS" if label == "NO TUMOR" else "FAIL"
    if label == "NO TUMOR": no_correct += 1
    no_total += 1
    print(f"  [{status}] {f:20s} -> {label:10s} ({conf}%)")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Tumor: {tumor_correct}/{tumor_total}")
print(f"No Tumor: {no_correct}/{no_total}")
total = tumor_total + no_total
correct = tumor_correct + no_correct
print(f"Overall Accuracy: {correct/total*100:.2f}%")
