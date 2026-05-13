import cv2
import matplotlib.pyplot as plt

# 1. Load the image using OpenCV
image = cv2.imread("cat_image.jpg")

# 2. Check if the image was loaded correctly
if image is None:
    print("Image could not be loaded. Check the file path.")
else:
    # 3. Print image information
    print("Image shape:", image.shape)
    print("Image data type:", image.dtype)

    # 4. Get dimensions and number of channels
    height, width, channels = image.shape

    print("Height:", height)
    print("Width:", width)
    print("Number of channels:", channels)

    # 5. Convert BGR image to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 6. Display the image
    plt.imshow(image_rgb)
    plt.title("Loaded Image")
    plt.axis("off")
    plt.show()