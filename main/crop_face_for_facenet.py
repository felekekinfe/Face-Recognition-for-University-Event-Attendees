import os
import cv2
from mtcnn import MTCNN

def crop_face(img_folder_path,cropped_output_path):
    detector = MTCNN()

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
    img_path = ''  
    cropped_output_path = ''
    crop_face(img_folder_path=img_path,cropped_output_path=cropped_output_path)
