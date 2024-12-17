import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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


# Run testing if needed
# test_knn()

if __name__ == "__main__":

    def knn(X_train, y_train, x_input, k=3):
        distances = []
        for i in range(len(X_train)):
            dist = euclidean_distance(X_train[i], x_input)
            distances.append((dist, y_train[i]))
        distances = sorted(distances)[:k]
        labels = [label for _, label in distances]
        return max(set(labels), key=labels.count)  # Majority vote

