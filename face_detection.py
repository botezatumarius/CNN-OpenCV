import cv2

def detect_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid image path.")
    
    # Convert the image to grayscale for better detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect face
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None  
    
    # Draw a red rectangle around the face
    x, y, w, h = faces[0]
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)  
    
    # Show the image with the rectangle
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return tuple(faces[0])

image_path = "images/0AA0A2.jpg"
face_coordinates = detect_face(image_path)

if face_coordinates:
    print(f"Face found at coordinates: {face_coordinates}")
else:
    print("No face detected in the image.")
