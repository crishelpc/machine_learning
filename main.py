import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from sklearn.decomposition import PCA

# Function to load images from folder with progress bar
def load_images_from_folder(folder):
    images = []
    labels = []
    print("Loading images...")
    for label in range(10):  # Assuming folders are named 0, 1, ..., 9
        path = os.path.join(folder, str(label))
        if not os.path.exists(path):
            print(f"Warning: Folder {path} does not exist. Skipping...")
            continue
        
        # Wrap the file list in tqdm to display a progress bar
        file_list = os.listdir(path)
        for filename in tqdm(file_list, desc=f"Loading label {label}", unit="file"):
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img_resized = cv2.resize(img, (28, 28))  # Resize to 28x28
                images.append(img_resized.flatten())    # Flatten to 1D array
                labels.append(label)
    return np.array(images), np.array(labels)

# Load or Train Function
# Load or Train Function
def load_or_train_data():
    train_file = "X_train.npy"
    labels_file = "y_train.npy"

    # Check if pre-existing training files exist
    if os.path.exists(train_file) and os.path.exists(labels_file):
        print("Loading training data from file...")
        X_train = np.load(train_file)
        y_train = np.load(labels_file)
        print("Training data loaded successfully.")
        return X_train, y_train

    # If files don't exist, train from scratch
    print("No training data found. Starting training...")
    X, y = load_images_from_folder("dataset")
    X = X / 255.0  # Normalize pixel values to range [0, 1]

    # Save the training data
    np.save(train_file, X)
    np.save(labels_file, y)
    print("Training data saved to file.")

    return X, y

# KNN functions (already provided)
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

# Example usage
X_train, y_train = load_or_train_data()  # Load or train the data

# For testing or prediction:
def test_knn():
    X, y = load_images_from_folder("dataset")  # Reload the dataset (this can be done once and skipped later)
    X = X / 255.0
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    correct = 0
    for i in range(len(X_test)):
        prediction = knn(X_train, y_train, X_test[i])
        if prediction == y_test[i]:
            correct += 1

    accuracy = correct / len(X_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # For KNN predictions
    predictions = [knn(X_train, y_train, x) for x in X_test]
    print("Test Accuracy:", accuracy_score(y_test, predictions))


if __name__ == "__main__":
# Run testing if needed
    test_knn()
