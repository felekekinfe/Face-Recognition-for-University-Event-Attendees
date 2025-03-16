import cv2
import time
import os
import numpy as np
from get_embedding import get_embedding
from load_embedding import load_embedding
from crop_face_for_facenet import batch_crop_face, verification_face_crop

def capture_and_verify_face(embedding_path):
    
    # Open the camera
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    if not cap.isOpened():
        print("Failed to open camera")
        return
    

    # Load Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    if face_cascade.empty():
        print("Error loading Haar cascade file")
        return None

    print("Looking for a face to verify...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Process if face is detected
        if len(faces) > 0:
            # Save temporary image
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Verify the face
            result = load_embedding_and_verify_image(temp_path, embedding_path)
            
            # Draw rectangle and result text
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                # Display result above the face
                cv2.putText(frame, result, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                           (0, 255, 0) if result == "Present" else (0, 0, 255), 2)
            
            # Show result for 3 seconds
            cv2.imshow("Camera Feed", frame)
            cv2.waitKey(3000)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            break

        # Display live feed
        cv2.imshow("Camera Feed", frame)

        # Exit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture cancelled by user")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

def load_embedding_and_verify_image(image_path=None, embedding_path=None):
    if image_path is None or not os.path.exists(image_path):
        print(f"Failed to load image")
        return "Error"

    # Crop image to 160x160
    img = verification_face_crop(image_path)  # returns verify_cropped_img path
    #img=image_path
    
    if not img or not os.path.exists(img):
        print("Failed to crop face")
        return "Error"

    # Get embedding
    claim_embedding = get_embedding(img)
    
    # Clean up cropped image
    if os.path.exists(img):
        os.remove(img)

    # Compare to known embeddings
    known_embeddings = load_embedding(embedding_path)
  
    
    
    if not known_embeddings:
        print("No known embeddings found")
        return "Error"

    min_dist = float('inf')
    for embedding in known_embeddings:
        dist = np.linalg.norm(claim_embedding - embedding)
        if dist < min_dist:
            min_dist = dist

    # Return result based on threshold
    print(min_dist)
    if min_dist < 0.7:
        return "Present"
    return "Not Present"
    

if __name__ == "__main__":
    
    # Specify your embedding path here
    embedding_path = "embeddings/embeddings.npy"
    # capture_and_verify_face(embedding_path)

    capture_and_verify_face(embedding_path)


