import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # Must be set before importing TF

import shutil
import cv2
import imutils
import numpy as np
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

import tf_keras as keras

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── Load the best model (from the notebook) ───
# The model files are HDF5 but saved without .h5 extension.
# Copy to a .h5 file so Keras can recognize the format.
MODEL_H5 = os.path.join(os.path.dirname(__file__), 'models', 'new_custom_model.h5')

best_model = keras.models.load_model(MODEL_H5)
IMG_WIDTH, IMG_HEIGHT = (224, 224)


# ─── Exact crop_brain_contour function from the notebook ───
def crop_brain_contour(image, plot=False):
    # Convert the image to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    return new_image


# ─── Preprocess a single image (exact steps from the notebook) ───
def preprocess_image(image_path):
    # load the image
    image = cv2.imread(image_path)
    # crop the brain and ignore the unnecessary rest part of the image
    image = crop_brain_contour(image, plot=False)
    # resize image
    image = cv2.resize(image, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    # normalize values
    image = image / 255.
    # reshape for model input
    image = np.expand_dims(image, axis=0)
    return image


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ─── Routes ───
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG.'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess (exact same pipeline as the notebook)
        image = preprocess_image(filepath)

        # Predict (exact same as the notebook)
        prediction_prob = best_model.predict(image)
        confidence = float(prediction_prob[0][0])

        if confidence >= 0.5:
            result = 'Tumor Detected'
            confidence_pct = round(confidence * 100, 2)
        else:
            result = 'No Tumor Detected'
            confidence_pct = round((1 - confidence) * 100, 2)

        return jsonify({
            'result': result,
            'confidence': confidence_pct,
            'image_url': f'/uploads/{filename}'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
