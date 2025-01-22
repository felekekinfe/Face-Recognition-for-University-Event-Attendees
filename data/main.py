import cv2
import pickle 
import numpy as np
import os
face_data=[]
name=input('enter name')
i=0
# Initialize the video capture
video = cv2.VideoCapture(0)

# Load the Haar Cascade for face detection
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = video.read()  # Read a frame from the video feed
    if not ret:
        break

    # Convert the frame to grayscale (face detection works better in grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        crop_img=frame[y:y+h,x:x+w]
        resize_img=cv2.resize(crop_img,(50,50))
        if len(face_data)<100 and i%2==0:
            face_data.append(resize_img)
        i+=1
        cv2.putText(frame, f"Faces: {len(faces)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Break the loop when 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

face_data=np.asarray(face_data)
face_data=face_data.reshape(100,-1)


if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl','wb') as f:
        pickle.dump(names,f)
else:
    with open('data/names.pkl','rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open('data/names.pkl') as f:
        pickle.dump(names,f)

if 'face_data.pkl' not in os.listdir('data/'):
    
    with open('data/face_data.pkl','wb') as f:
        pickle.dump(face_data,f)
else:
    with open('data/face_data.pkl','rb') as f:
        faces=pickle.load(f)
    faces=face_data.append(faces)
    with open('data/face_data.pkl') as f:
        pickle.dump(names,f)
        