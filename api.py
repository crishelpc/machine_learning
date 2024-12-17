from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import os
import cv2
from sklearn.model_selection import train_test_split
from main import load_images_from_folder, load_or_train_data, euclidean_distance, knn

app = Flask(__name__)

# # Load the training data
X_train, y_train = load_or_train_data()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Open the image
        image = Image.open(file).convert("L")  # Convert to grayscale
        img_resized = image.resize((28, 28))   # Resize to 28x28
        img_inverted = ImageOps.invert(img_resized)  # Invert colors (white digits on black)
        img_array = np.array(img_inverted)

        # Apply binary thresholding
        _, img_thresh = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY_INV)

        # Check for multiple digits using contours
        contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            return jsonify({"error": "Please draw a single digit only!"}), 400

        # Flatten and normalize the image
        img_flattened = img_array.flatten() / 255.0

        # Predict using the KNN model
        prediction = knn(X_train, y_train, img_flattened)

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == "__main__":
    app.run(debug=True)
