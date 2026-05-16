# exercise7.py

import torch
import matplotlib.pyplot as plt
from exercise6 import model, test_loader, device  

# Ensure model is in evaluation mode
model.eval()

# Get one batch of test images
dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)

# Make predictions
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Visualize first 12 images
plt.figure(figsize=(12,6))
for i in range(12):
    img = images[i].cpu().squeeze()  # move to CPU for matplotlib
    true_label = labels[i].item()
    pred_label = predicted[i].item()
    correct = true_label == pred_label

    plt.subplot(3,4,i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Pred: {pred_label}\nTrue: {true_label}", color='green' if correct else 'red')

plt.tight_layout()
plt.show()