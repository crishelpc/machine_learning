from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import os
import cv2
from sklearn.model_selection import train_test_split
from main import knn, load_or_train_data

app = Flask(__name__)

# Load the training data
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
        image = Image.open(file).convert("L")
        img_resized = image.resize((28, 28))
        img_inverted = ImageOps.invert(img_resized)  # Invert colors (white digits on black)
        img_array = np.array(img_inverted).flatten() / 255.0

        # Predict using the KNN model
        prediction = knn(X_train, y_train, img_array)

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
