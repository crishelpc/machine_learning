import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.model_selection import train_test_split
import os
import pickle

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
    # Try loading pre-trained data
    try:
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

# GUI with Drawing Canvas
class KNNDrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("KNN Digit Classifier - Draw & Predict")
        self.root.geometry("400x500")

        self.canvas = tk.Canvas(root, bg="black", width=280, height=280)
        self.canvas.pack(pady=10)

        # Bind mouse events to the canvas
        self.canvas.bind("<B1-Motion>", self.paint)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(pady=5)

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.pack(pady=5)

        self.result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 14))
        self.result_label.pack(pady=10)

        # Initialize a PIL image to store the drawing
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)

        # Load or train the data
        self.X_train, self.y_train = load_or_train_data()  # Load or train the data

    def paint(self, event):
        x, y = event.x, event.y
        r = 8  # Brush radius
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction: ")

    def predict_digit(self):
        # Resize the drawn image to 28x28 and normalize
        img_resized = self.image.resize((28, 28))
        img_inverted = ImageOps.invert(img_resized)  # Invert colors: white digits on black
        img_array = np.array(img_inverted).flatten() / 255.0

        # Use the KNN model to predict
        prediction = knn(self.X_train, self.y_train, img_array)
        self.result_label.config(text=f"Prediction: {prediction}")


# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = KNNDrawApp(root)
    root.mainloop()
