import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# --- Load and Preprocess the Dataset ---
dataset_path = "dataset/"  # Change this to your dataset path
img_size = 28  # Resize images to 28x28

def load_data(dataset_path):
    X = []  # Features (images)
    y = []  # Labels (0-9)

    for label in range(10):  # Digits 0-9
        folder_path = os.path.join(dataset_path, str(label))
        for file_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, file_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            img = cv2.resize(img, (img_size, img_size))       # Resize to 28x28
            img = img.flatten() / 255.0                      # Normalize to [0, 1]
            X.append(img)
            y.append(label)

    return np.array(X), np.array(y)

# Load dataset
print("Loading dataset...")
X, y = load_data(dataset_path)
print(f"Dataset loaded: {X.shape[0]} images")

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]} images")
print(f"Testing set size: {X_test.shape[0]} images")

# --- Neural Network Implementation ---
# Parameters
input_size = img_size * img_size  # 784 (28x28 pixels)
hidden_size_1 = 128               # First hidden layer size
hidden_size_2 = 64                # Second hidden layer size
output_size = 10                  # 10 digits (0-9)
learning_rate = 0.01              # Learning rate
epochs = 50                       # Number of training epochs
batch_size = 64                   # Mini-batch size
lambda_ = 0.01                    # Regularization strength

# Initialize weights with He Initialization
W1 = np.random.randn(hidden_size_1, input_size) * np.sqrt(2 / input_size)
b1 = np.zeros((hidden_size_1, 1))
W2 = np.random.randn(hidden_size_2, hidden_size_1) * np.sqrt(2 / hidden_size_1)
b2 = np.zeros((hidden_size_2, 1))
W3 = np.random.randn(output_size, hidden_size_2) * np.sqrt(2 / hidden_size_2)
b3 = np.zeros((output_size, 1))

# Activation functions
def leaky_relu(Z, alpha=0.01):
    return np.maximum(alpha * Z, Z)

def leaky_relu_derivative(Z, alpha=0.01):
    dZ = np.ones_like(Z)
    dZ[Z < 0] = alpha
    return dZ

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

# Forward propagation
def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    Z1 = np.dot(W1, X.T) + b1
    A1 = leaky_relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = leaky_relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# Loss function with L2 regularization
def compute_loss(Y, A3, W1, W2, W3, lambda_):
    m = Y.shape[0]
    epsilon = 1e-8  # Prevent log(0)
    log_probs = -np.log(np.clip(A3[Y, range(m)], epsilon, 1 - epsilon))
    loss = np.sum(log_probs) / m
    l2_reg = (lambda_ / (2 * m)) * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    return loss + l2_reg

# Backward propagation
def backward_propagation(X, Y, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3):
    m = X.shape[0]
    Y_one_hot = np.eye(output_size)[Y].T  # One-hot encoding
    
    dZ3 = A3 - Y_one_hot
    dW3 = (1 / m) * np.dot(dZ3, A2.T) + (lambda_ / m) * W3
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * leaky_relu_derivative(Z2)
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + (lambda_ / m) * W2
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * leaky_relu_derivative(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X) + (lambda_ / m) * W1
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    
    return dW1, db1, dW2, db2, dW3, db3

# Mini-batch gradient descent training
print("Training the neural network...")
num_batches = X_train.shape[0] // batch_size

for epoch in range(epochs):
    for i in range(num_batches):
        # Get mini-batch
        X_batch = X_train[i * batch_size:(i + 1) * batch_size]
        y_batch = y_train[i * batch_size:(i + 1) * batch_size]
        
        # Forward propagation
        Z1, A1, Z2, A2, Z3, A3 = forward_propagation(X_batch, W1, b1, W2, b2, W3, b3)
        
        # Backward propagation
        dW1, db1, dW2, db2, dW3, db3 = backward_propagation(X_batch, y_batch, Z1, A1, Z2, A2, Z3, A3, W1, W2, W3)
        
        # Update parameters
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W3 -= learning_rate * dW3
        b3 -= learning_rate * db3

    # Compute loss for the entire training set
    _, _, _, _, _, A3 = forward_propagation(X_train, W1, b1, W2, b2, W3, b3)
    loss = compute_loss(y_train, A3, W1, W2, W3, lambda_)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

print("Training complete!")

# --- Testing and Evaluation ---
def predict(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_propagation(X, W1, b1, W2, b2, W3, b3)
    return np.argmax(A3, axis=0)

print("Evaluating on test data...")
predictions = predict(X_test, W1, b1, W2, b2, W3, b3)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

np.save("W1.npy", W1)
np.save("b1.npy", b1)
np.save("W2.npy", W2)
np.save("b2.npy", b2)
