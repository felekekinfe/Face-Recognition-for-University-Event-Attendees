import os
import cv2
from mtcnn import MTCNN

detector = MTCNN()

'''verification_face_crop function crop a single image we want to verify to 160X160 and return its path'''
def verification_face_crop(verify_img=None, verify_cropped_path=None):
    if verify_img is not None:
        img_name = os.path.basename(verify_img)
        
        # Default path for cropped images
        if verify_cropped_path is None:
            verify_cropped_path = 'cropped_verify'
            os.makedirs(verify_cropped_path, exist_ok=True)

        output_path = os.path.join(verify_cropped_path, img_name)

        # Debugging: Check if the image can be loaded
        print(f"Attempting to load image from {verify_img}")
        verify_img = cv2.imread(verify_img)
        
        if verify_img is None:
            print(f"Failed to load image {verify_img}. Check if the path is correct.")
            return None
        
        # Assuming detect_faces() is the function that detects faces in the image
        faces = detector.detect_faces(verify_img)
        
        # Debugging: Check if any faces are detected
        if len(faces) == 0:
            print("No faces detected.")
            return None
        
        for face in faces:
            # Extract face bounding box
            x, y, w, h = face['box']
            face_img = verify_img[y:y + h, x:x + w]  # Crop the face
            
            # Resize the cropped face to 160x160
            face_img = cv2.resize(face_img, (160, 160))
            
            # Debugging: Check if face is cropped and resized correctly
            if face_img is not None and face_img.shape[0] == 160 and face_img.shape[1] == 160:
                print(f"Face cropped and resized to {face_img.shape}")
            else:
                print("Error cropping or resizing the face.")

            # Write the cropped face to the output path
            if cv2.imwrite(output_path, face_img):
                print(f"Face successfully saved to {output_path}")
                return output_path
            else:
                print(f"Failed to save the cropped face to {output_path}")
                return None

    else:
        print("No image provided.")
        return None
'''  batch_crop_face function used to crop face from a given folder to make an embedding of each image which used for verification purpose'''
def batch_crop_face(img_folder_path=None,cropped_output_path=None):
    
    if img_folder_path is None:
    	return 'failed to load image'
                
    if not os.path.exists(cropped_output_path):
        os.makedirs(cropped_output_path)


    for img_file in os.listdir(img_folder_path):
        input_img = os.path.join(img_folder_path, img_file)

        # if not os.path.isdir(input_img):
        #     continue

        img = cv2.imread(input_img)

        if img is None:
            print(f"Skipping {img} as it cannot be read.")
            continue

        # Detect faces using MTCNN
        faces = detector.detect_faces(img)

        # If faces are detected, crop and save them
        if len(faces) > 0:
            for i, face in enumerate(faces):
                x, y, w, h = face['box']  # Get the face coordinates
                face_img = img[y:y + h, x:x + w]  # Crop the face
                face_img = cv2.resize(face_img, (160, 160))  # Resize the face to 160x160

                
                output_path = os.path.join(cropped_output_path, f"{img_file.split('.')[0]}_face{i}.jpg")
                cv2.imwrite(output_path, face_img)
        else:
            print(f"No faces detected in {img_file}, skipping.")

    print('Face cropping process completed.')
if __name__=='__main__':
    img_path = 'test'  
    cropped_output_path = 'cropped_test'
    batch_crop_face(img_folder_path=img_path,cropped_output_path=cropped_output_path)
