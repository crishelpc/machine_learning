from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image, ImageOps
import base64
import cv2
import io
import re
from sklearn.model_selection import train_test_split
import os

app = Flask(__name__)

# KNN Functions
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn(X_train, y_train, x_input, k=3):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_input)
        distances.append((dist, y_train[i]))
    distances = sorted(distances)[:k]
    labels = [label for _, label in distances]
    return max(set(labels), key=labels.count)  # Majority vote

# Function to load images
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in range(10):  # Assuming folders are named 0, 1, ..., 9
        path = os.path.join(folder, str(label))
        for filename in os.listdir(path):
            img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (28, 28))  # Resize to 28x28
                images.append(img_resized.flatten())    # Flatten to 1D array
                labels.append(label)
    return np.array(images), np.array(labels)

# Load or Train Function
def load_or_train_data():
    try:
        # Try loading pre-trained data
        X_train = np.load("X_train.npy")
        y_train = np.load("y_train.npy")
        print("Loaded training data from file.")
    except FileNotFoundError:
        # If files don't exist, train from scratch
        print("Training data not found, training now...")
        X, y = load_images_from_folder("dataset")
        X = X / 255.0  # Normalize
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Save training data to avoid retraining in future
        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)
        print("Training data saved to file.")

    return X_train, y_train

# Load the training data
X_train, y_train = load_or_train_data()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json.get("imageData")
    if not data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode the base64 image
    image_data = re.sub("^data:image/png;base64,", "", data)
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert("L")
    img_resized = image.resize((28, 28))
    img_inverted = ImageOps.invert(img_resized)  # Invert colors (white digits on black)
    img_array = np.array(img_inverted).flatten() / 255.0

    # Predict using the KNN model
    prediction = knn(X_train, y_train, img_array)

    return jsonify({"prediction": int(prediction)})

if __name__ == "__main__":
    app.run(debug=True)