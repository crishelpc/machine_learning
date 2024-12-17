import cv2
import numpy as np

# Load the image
image = cv2.imread('dataset/0/0.png', cv2.IMREAD_GRAYSCALE)

# Apply Histogram Equalization
image_eq = cv2.equalizeHist(image)

# OR: Use CLAHE for adaptive contrast enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
image_clahe = clahe.apply(image)

# Save the output to check the results
cv2.imwrite('image_eq.png', image_eq)
cv2.imwrite('image_clahe.png', image_clahe)

# Show the images
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(image_eq, cmap='gray')
plt.title("Histogram Equalized")

plt.subplot(1, 3, 3)
plt.imshow(image_clahe, cmap='gray')
plt.title("CLAHE Enhanced")

plt.show()
