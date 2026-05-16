import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Current folder (where exercise4.py is)
dataset_path = "."  # LAB-10 folder

# Names of the category folders
categories = ["cats", "dogs"]

# Initialize feature matrix X and label vector Y
X = []
Y = []

# Loop through each category
for label, category in enumerate(categories):
    folder_path = os.path.join(dataset_path, category)
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
    
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)
        if image is not None:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Resize to fixed size (100x100)
            resized = cv2.resize(gray, (100, 100))
            # Flatten the image and add to feature matrix
            X.append(resized.flatten())
            # Add label (0 for cats, 1 for dogs)
            Y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
Y = np.array(Y)
    
# Print shapes
print(f"Feature matrix X shape: {X.shape}")
print(f"Label vector Y shape: {Y.shape}")

# Display first 4 images as a sanity check
plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.imshow(X[i].reshape(100, 100), cmap="gray")
    plt.title(f"Label: {Y[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()