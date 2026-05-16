
# Import the dataset from exercise4
from exercise4 import X, Y  

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# SPLIT DATASET
# 20% of data used for testing

X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=42)

# CREATE AND TRAIN KNN

knn = KNeighborsClassifier(n_neighbors=3)  # k = 3
knn.fit(X_train, Y_train)


# PREDICT

Y_pred = knn.predict(X_test)

# EVALUATE
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Classification Accuracy: {accuracy*100:.2f}%")

# CONFUSION MATRIX
cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Cat","Dog"], yticklabels=["Cat","Dog"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# Show misclassified examples

misclassified_idx = np.where(Y_test != Y_pred)[0]  # indices where prediction != actual

print(f"Number of misclassified images: {len(misclassified_idx)}")

plt.figure(figsize=(12,6))
for i, idx in enumerate(misclassified_idx[:12]):  # show first 12 misclassified
    img = X_test[idx].reshape(100,100)
    plt.subplot(3,4,i+1)
    plt.imshow(img, cmap="gray")
    plt.title(f"True: {Y_test[idx]}, Pred: {Y_pred[idx]}")
    plt.axis("off")
plt.tight_layout()
plt.show()

