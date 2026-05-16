# exercise6.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. Device configuration (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(torch.cuda.get_device_name(0))



# 2. Dataset and DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),           # convert images to PyTorch tensors
    transforms.Normalize((0.5,), (0.5,))  # normalize to [-1, 1]
])

# Download MNIST dataset
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 3. Define CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, 1)   # input 1 channel (grayscale), 16 filters, 3x3 kernel
        self.pool = nn.MaxPool2d(2, 2)        # 2x2 max pooling
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # 16 input, 32 filters
        self.fc1 = nn.Linear(32*5*5, 128)     # fully connected layer
        self.fc2 = nn.Linear(128, 10)         # output 10 classes (digits 0-9)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 32*5*5)   # flatten feature maps
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN().to(device)

# 4. Loss and Optimizer

criterion = nn.CrossEntropyLoss()          # multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 5. Train the Model

num_epochs = 5

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # move data to GPU
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")


# 6. Evaluate the Model

correct = 0
total = 0
model.eval()  # set model to evaluation mode
with torch.no_grad():  # disable gradient computation
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


# 7. Visualize Some Predictions

dataiter = iter(test_loader)
images, labels = next(dataiter)
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)



plt.figure(figsize=(12,4))
for i in range(8):
    plt.subplot(2,4,i+1)
    plt.imshow(images[i].cpu().squeeze(), cmap='gray')  # move to CPU for plotting
    plt.title(f"Pred: {predicted[i].item()}, True: {labels[i].item()}")
    plt.axis('off')
plt.show()
