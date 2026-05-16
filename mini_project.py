# mini_project_cats_dogs.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import cv2  # For image preprocessing

# 1. Dataset Loading and Preprocessing

image_size = 100  # resize to 100x100
X = []
Y = []

folders = ["cats", "dogs"]
for label, folder in enumerate(folders):
    folder_path = os.path.join(os.getcwd(), folder)
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (image_size, image_size))
        X.append(resized.flatten())  # Flatten for classical ML
        Y.append(label)

X = np.array(X)
Y = np.array(Y)

print("Feature matrix X shape:", X.shape)
print("Label vector Y shape:", Y.shape)

# Split data for classical ML
X_train_ml, X_test_ml, Y_train_ml, Y_test_ml = train_test_split(X, Y, test_size=0.2, random_state=42)

# ================================
# 2. Classical ML: KNN
# ================================
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_ml, Y_train_ml)
Y_pred_knn = knn.predict(X_test_ml)
acc_knn = accuracy_score(Y_test_ml, Y_pred_knn)
print(f"KNN Accuracy: {acc_knn*100:.2f}%")

# Confusion matrix KNN
cm_knn = confusion_matrix(Y_test_ml, Y_pred_knn)
plt.figure(figsize=(5,4))
sns.heatmap(cm_knn, annot=True, fmt="d", cmap="Blues",
            xticklabels=folders, yticklabels=folders)
plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================================
# 3. Classical ML: Logistic Regression
# ================================
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_ml, Y_train_ml)
Y_pred_log = logreg.predict(X_test_ml)
acc_log = accuracy_score(Y_test_ml, Y_pred_log)
print(f"Logistic Regression Accuracy: {acc_log*100:.2f}%")

# Confusion matrix Logistic Regression
cm_log = confusion_matrix(Y_test_ml, Y_pred_log)
plt.figure(figsize=(5,4))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Greens",
            xticklabels=folders, yticklabels=folders)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================================
# 4. Prepare Data for CNN
# ================================
# Reshape to (N, 1, H, W) for PyTorch CNN
X_cnn = np.array([img.reshape(1, image_size, image_size) for img in X])
Y_cnn = np.array(Y)

# Split
X_train_cnn, X_test_cnn, Y_train_cnn, Y_test_cnn = train_test_split(X_cnn, Y_cnn, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_cnn = torch.tensor(X_train_cnn, dtype=torch.float32)
X_test_cnn = torch.tensor(X_test_cnn, dtype=torch.float32)
Y_train_cnn = torch.tensor(Y_train_cnn, dtype=torch.long)
Y_test_cnn = torch.tensor(Y_test_cnn, dtype=torch.long)

# DataLoaders
train_loader = DataLoader(TensorDataset(X_train_cnn, Y_train_cnn), batch_size=8, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_cnn, Y_test_cnn), batch_size=8, shuffle=False)

# ================================
# 5. Define Simple CNN
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32*23*23, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 classes: cat/dog

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32*23*23)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ================================
# 6. Train CNN
# ================================
epochs = 5
for epoch in range(epochs):
    running_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# ================================
# 7. Evaluate CNN
# ================================
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc_cnn = np.mean(np.array(all_preds) == np.array(all_labels))
print(f"CNN Accuracy: {acc_cnn*100:.2f}%")

# Confusion matrix CNN
cm_cnn = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(5,4))
sns.heatmap(cm_cnn, annot=True, fmt="d", cmap="Oranges",
            xticklabels=folders, yticklabels=folders)
plt.title("CNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ================================
# 8. Visualize Misclassified Images (CNN)
# ================================
mis_idx = [i for i, (a,p) in enumerate(zip(all_labels, all_preds)) if a != p]
plt.figure(figsize=(12,6))
for i, idx in enumerate(mis_idx[:12]):
    img = X_test_cnn[idx].squeeze().numpy()
    true_label = all_labels[idx]
    pred_label = all_preds[idx]
    plt.subplot(3,4,i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {folders[true_label]}, Pred: {folders[pred_label]}", color='red')
    plt.axis('off')
plt.tight_layout()
plt.show()