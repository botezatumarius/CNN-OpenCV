import cv2
import numpy as np
import random
import os
import matplotlib.pyplot as plt

# Path to the folder containing images
images_folder = 'images'

# Function to get a random image from the folder
def get_random_image(folder):
    image_files = [file for file in os.listdir(folder) if file.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError("No image files found in the specified folder.")
    random_image = random.choice(image_files)
    return os.path.join(folder, random_image)

def blur_image(image, kernel_size=(15, 15)):
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred

def sharpen_image(image):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

def main():
    image_path = get_random_image(images_folder)
    image = cv2.imread(image_path)
    
    # Convert BGR image to RGB for Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    blurred_image = blur_image(image_rgb)
    sharpened_image = sharpen_image(image_rgb)

    # Plotting the images
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(blurred_image)
    plt.title("Blurred Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sharpened_image)
    plt.title("Sharpened Image")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
