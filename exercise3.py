import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image using OpenCV
image = cv2.imread("cat_image.jpg")

# Check if the image is loaded correctly
if image is None:
    print("Image could not be loaded. Check the file path.")
else:
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Compute the pixel intensity histogram
    hist = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])

    # 2. Calculate the mean pixel value
    mean_pixel_value = np.mean(grayscale_image)
    
    # 3. Calculate the standard deviation of pixel intensities
    std_dev_pixel_value = np.std(grayscale_image)

    # Print the results
    print(f"Mean Pixel Value: {mean_pixel_value}")
    print(f"Standard Deviation of Pixel Intensities: {std_dev_pixel_value}")

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    
    # Plot the pixel intensity histogram
    plt.subplot(1, 2, 1)
    plt.plot(hist)
    plt.title("Pixel Intensity Histogram")
    plt.xlabel("Pixel Intensity (0-255)")
    plt.ylabel("Frequency")
    
    # Display the grayscale image
    plt.subplot(1, 2, 2)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()