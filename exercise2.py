import cv2
import matplotlib.pyplot as plt

# Load the image using OpenCV
image = cv2.imread("cat_image.jpg")

# Check if the image is loaded correctly
if image is None:
    print("Image could not be loaded. Check the file path.")
else:
    # 1. Convert to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Resize the image to a fixed resolution (e.g., 500x500)
    resized_image = cv2.resize(grayscale_image, (500, 500))

    # 3. Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(resized_image, (5, 5), 0)

    # 4. Detect edges using Canny algorithm
    edges_image = cv2.Canny(blurred_image, threshold1=50, threshold2=150)

    # Plot the images
    plt.figure(figsize=(12, 12))

    # Display the grayscale image
    plt.subplot(2, 2, 1)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis("off")

    # Display the resized image
    plt.subplot(2, 2, 2)
    plt.imshow(resized_image, cmap='gray')
    plt.title("Resized Image")
    plt.axis("off")

    # Display the blurred image
    plt.subplot(2, 2, 3)
    plt.imshow(blurred_image, cmap='gray')
    plt.title("Gaussian Blurred Image")
    plt.axis("off")

    # Display the edges image
    plt.subplot(2, 2, 4)
    plt.imshow(edges_image, cmap='gray')
    plt.title("Edges Detected (Canny)")
    plt.axis("off")

    # Show all images
    plt.tight_layout()
    plt.show()