import cv2
import numpy as np

def find_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def check_passport_criteria(image_path, show_result=False):
    image = cv2.imread(image_path)
    
    if image is None:
        raise ValueError("Image not found or invalid image path.")
    
    # Check if the image is grayscale
    gray = True
    for row in image:
        for pixel in row:
            r, g, b = pixel[0], pixel[1], pixel[2]
            if r != g or r != b or g != b:
                gray = False
                break
        if not gray:
            break
    
    if gray:
        if show_result:
            print(f"The image is grayscale with RGB values: {image[0][0]}")
        return False
    
    # Check if the image is in portrait or square orientation
    height, width, _ = image.shape
    if height < width:
        if show_result:
            print(f"The image is landscape with dimensions: {height}x{width}")
        return False

    # Detect eyes and check if they are at the same level
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_placements = eye_cascade.detectMultiScale(gray_image)
    
    if len(eye_placements) != 2:
        if show_result:
            print(f"Found {len(eye_placements)} eyes")
        return False
    
    eye1_x, eye1_y, eye1_w, eye1_h = eye_placements[0]
    eye2_x, eye2_y, eye2_w, eye2_h = eye_placements[1]
    
    if abs(eye1_y - eye2_y) > 5:
        if show_result:
            print(f"Eye placement deviation is too high: {eye1_y} - {eye2_y}")
        return False
    
    # Ensure there is exactly one face in the image
    faces = find_faces(image)
    if faces is None or len(faces) != 1:
        if show_result:
            print(f"Found {len(faces)} faces")
        return False

    # Ensure the face occupies between 20% to 50% of the image area
    face_x, face_y, face_w, face_h = faces[0]
    face_area = face_w * face_h
    image_area = height * width
    
    if face_area < 0.2 * image_area or face_area > 0.5 * image_area:
        if show_result:
            print(f"Face area is {face_area} and image area is {image_area}")
        return False

    # If all checks pass, return True
    return True

import os

image_folder = "images"  # Replace with the path to your image folder
for filename in os.listdir(image_folder):
    image_path = os.path.join(image_folder, filename)
    
    if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):  # Process image files only
        result = check_passport_criteria(image_path, show_result=True)
        print(f"{filename}: {'Passes' if result else 'Does not pass'}")
