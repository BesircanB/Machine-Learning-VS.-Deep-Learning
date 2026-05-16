# Machine Learning VS. Deep Learning

This repository contains exercises and a mini-project for exploring classical machine learning and deep learning techniques for image processing and classification.

---

## **Table of Contents**

1. [Overview](#overview)  
2. [Setup Instructions](#setup-instructions)  
3. [Exercises](#exercises)  
4. [Mini Project](#mini-project)  
5. [Results](#results)  


---

## **Overview**

This repository includes step-by-step exercises demonstrating key concepts in:

- Image preprocessing  
- Feature extraction  
- Classical machine learning (KNN, Logistic Regression)  
- Convolutional Neural Networks (CNNs) using PyTorch  
- Model evaluation and visualization  

The final **mini-project** compares classical ML and CNN approaches on a Cats vs Dogs image dataset.

---

## **Setup Instructions**

1. Clone the repository:

```bash
git clone https://github.com/YourUsername/Machine-Learning-VS-Deep-Learning.git
cd Machine-Learning-VS-Deep-Learning
```

2. Create a virtual environment:
```
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

3.Install dependencies
```
pip install -r requirements.txt
```

4.Ensure your dataset is in the cats/ and dogs/ folders. You can download from Kaggle.

# Exercises
## Exercise 1 – Basic Image Loading
Load and display an image using OpenCV.
Understand image channels and pixel intensity.

## Exercise 2 – Basic Image Processing
Convert images to grayscale.
Resize images.
Apply Gaussian blur.
Detect edges with Canny algorithm.
## Exercise 3 – Image Features
Compute pixel intensity histogram.
Compute mean and standard deviation of pixel intensities.
Understand how numerical features represent images.
## Exercise 4 – Image Dataset Preparation
Load multiple images from folders.
Convert to grayscale and resize.
Store as NumPy arrays for feature matrix X and label vector Y.
## Exercise 5 – Classical Image Classification
Train K-Nearest Neighbors (KNN) and Logistic Regression classifiers.
Evaluate performance using accuracy and confusion matrices.
Compare classical ML performance.
## Exercise 6 – Convolutional Neural Networks
Build a simple CNN using PyTorch.
Train CNN on image dataset.
Evaluate accuracy.
Visualize predictions.
## Exercise 7 – Visualization of Results
Display test images with predicted and true labels.
Highlight correct vs incorrect predictions.
Discuss why CNNs are effective for image analysis.


## Mini Project – Cats vs Dogs Classification
Dataset: 500 cat images and 500 dog images, resized to 100x100 pixels.
### Preprocessing:
- Grayscale conversion
- Resizing
- Flattening for classical ML
- Normalization for CNN
- Data augmentation for CNN
### Classical ML:
- KNN (k=3)
- Logistic Regression
### CNN:
- 2 convolutional layers, max pooling, fully connected layer, dropout
- Trained on GPU using PyTorch
#### Evaluation:
- Accuracy scores
- Confusion matrices
- Visualization of misclassified images
#### Purpose:
- Compare classical ML and CNN performance
- Show the benefit of convolutional networks for spatial feature learning
#### Results
- KNN Accuracy: 54%
- Logistic Regression Accuracy: 50%
- CNN Accuracy: 57%