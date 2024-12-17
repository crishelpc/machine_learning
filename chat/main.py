import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageDraw

# --- Neural Network Weights ---
# Replace these with your trained weights and biases
W1 = np.load("W1.npy")  # Pre-saved weights
b1 = np.load("b1.npy")
W2 = np.load("W2.npy")
b2 = np.load("b2.npy")

# --- Neural Network Functions ---
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X.T) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return A2

def predict(image, W1, b1, W2, b2):
    A2 = forward_propagation(image, W1, b1, W2, b2)
    return np.argmax(A2, axis=0)

# --- GUI with Canvas ---
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")
        
        # Canvas settings
        self.canvas_width = 280
        self.canvas_height = 280
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack()
        
        # Image and drawing
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)

        # Buttons
        self.predict_button = tk.Button(root, text="Predict", command=self.predict_number)
        self.predict_button.pack()
        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack()
        
        # Bind mouse events
        self.canvas.bind("<B1-Motion>", self.draw_line)

    def draw_line(self, event):
        x, y = event.x, event.y
        r = 10  # Brush size
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='white', outline='white')
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)

    def preprocess_image(self):
        # Resize to 28x28 and normalize
        img = self.image.resize((28, 28)).convert('L')
        img_array = np.array(img).flatten() / 255.0
        return img_array.reshape(1, -1)

    def predict_number(self):
        img = self.preprocess_image()
        prediction = predict(img, W1, b1, W2, b2)
        result_label.config(text=f"Predicted Number: {prediction}")

# --- Main App ---
if __name__ == "__main__":
    # Load saved weights
    W1 = np.load("W1.npy")
    b1 = np.load("b1.npy")
    W2 = np.load("W2.npy")
    b2 = np.load("b2.npy")

    # Create the GUI
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    result_label = tk.Label(root, text="Predicted Number: None", font=("Helvetica", 20))
    result_label.pack()
    root.mainloop()
