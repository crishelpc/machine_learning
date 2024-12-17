import os
import numpy as np
import cv2  # Use OpenCV for image reading and processing
from Model import neural_network
from RandInitialize import initialise
from Prediction import predict
from scipy.optimize import minimize

# Function to load images from folder structure
def load_images_from_folders(root_dir, image_size=(28, 28)):
    X = []  # Feature matrix
    y = []  # Labels
    for label in range(10):  # Assuming folders are named 0 to 9
        folder_path = os.path.join(root_dir, str(label))
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if file_path.endswith(('.png', '.jpg', '.jpeg')):  # Supported image formats
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read image in grayscale
                if image is None:
                    print(f"Failed to load {file_path}")
                    continue
                image = cv2.resize(image, image_size)  # Resize
                X.append(image.flatten())  # Flatten to 1D
                y.append(label)
    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {X.shape[0]} images with labels.")
    return X, y

# Load dataset from folders
root_directory = 'dataset/'  # Replace with the path to your dataset
X, y = load_images_from_folders(root_directory)

# Normalize pixel values to [0, 1]
X = X / 255.0

# Split into training (60,000) and testing (10,000) sets
split_idx = 60000
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Model parameters
m = X.shape[0]
input_layer_size = 784  # 28x28 images
hidden_layer_size = 256
num_labels = 10  # Classes [0-9]

# Initialize Thetas randomly
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)

# Unroll parameters
initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
maxiter = 300
lambda_reg = 0.01  # Regularization

# Minimize cost function to train weights
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)
results = minimize(neural_network, x0=initial_nn_params, args=myargs,
                   options={'disp': True, 'maxiter': maxiter}, method="L-BFGS-B", jac=True)

nn_params = results["x"]

# Reshape trained weights
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], 
                    (hidden_layer_size, input_layer_size + 1))
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
                    (num_labels, hidden_layer_size + 1))

# Test accuracy
pred_test = predict(Theta1, Theta2, X_test)
print('Test Set Accuracy: {:.2f}%'.format(np.mean(pred_test == y_test) * 100))

# Training accuracy
pred_train = predict(Theta1, Theta2, X_train)
print('Training Set Accuracy: {:.2f}%'.format(np.mean(pred_train == y_train) * 100))

# Precision evaluation
true_positive = np.sum(pred_train == y_train)
false_positive = len(y_train) - true_positive
precision = true_positive / (true_positive + false_positive)
print('Precision = {:.4f}'.format(precision))

# Save trained weights
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')
