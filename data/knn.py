from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os

# Initialize the video capture
video = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load face data and labels
with open('data/names.pkl', 'rb') as f:
    labels = pickle.load(f)
with open('data/face_data.pkl', 'rb') as f:
    faces = pickle.load(f)

print(f"Faces shape: {faces.shape}")
print(f"Number of labels: {len(labels)}")

# Train the kNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
faces_normalized = faces / 255.0  # Normalize pixel values to range [0, 1]
knn.fit(faces_normalized, labels)

while True:
    ret, frame = video.read()  # Read a frame from the video feed
    if not ret:
        break

    # Convert the frame to grayscale (face detection works better in grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    detected_faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Process each detected face
    for (x, y, w, h) in detected_faces:
        # Extract and preprocess the face
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (50, 50))  # Resize to match training data
        resized_img = resized_img.flatten() / 255.0  # Flatten and normalize to range [0, 1]
        resized_img = resized_img.reshape(1, -1)  # Reshape to (1, 2500)

        # Make a prediction
        prediction = knn.predict(resized_img)[0]

        # Display the prediction on the frame
        cv2.putText(
            frame, 
            prediction,  # Display the predicted label
            (x, y - 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
