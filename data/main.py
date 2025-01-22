import cv2
import pickle
import numpy as np
import os
import time

# Initialize variables
face_data = []
name = input('Enter name: ')
i = 0

# Create the data directory if it doesn't exist
if not os.path.exists('data/'):
    os.makedirs('data/')

# Initialize the video capture
video = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Timer for data collection frequency
start_time = time.time()

while True:
    ret, frame = video.read()  # Read a frame from the video feed
    if not ret:
        break

    # Convert the frame to grayscale (face detection works better in grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Process each detected face
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resize_img = cv2.resize(crop_img, (50, 50))  # Resize to a fixed size

        # Collect up to 100 samples with a 0.2s interval
        if len(face_data) < 100 and (time.time() - start_time) > 0.2:
            face_data.append(resize_img.flatten())  # Flatten the image and append
            start_time = time.time()

        # Display the count on the frame
        cv2.putText(frame, f"Samples: {len(face_data)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop when 'q' is pressed or data collection is complete
    if cv2.waitKey(1) & 0xFF == ord('q') or len(face_data) >= 100:
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

# Convert face data to NumPy array and normalize
face_data = np.asarray(face_data) / 255.0  # Normalize pixel values
face_data = face_data.reshape(100, -1)

# Save the face data and labels
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    if name in names:
        print(f"Warning: {name} already exists in the dataset.")
    names += [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'face_data.pkl' not in os.listdir('data/'):
    with open('data/face_data.pkl', 'wb') as f:
        pickle.dump(face_data, f)
else:
    with open('data/face_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    if faces.shape[1] == face_data.shape[1]:  # Ensure same number of features
        faces = np.vstack([faces, face_data])  # Stack the new data vertically
        with open('data/face_data.pkl', 'wb') as f:
            pickle.dump(faces, f)
    else:
        print("Error: Face data dimensions do not match!")
