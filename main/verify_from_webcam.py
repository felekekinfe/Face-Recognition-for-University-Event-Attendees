
# import cv2
# import os
# import numpy as np
# from get_embedding import get_embedding
# from load_embedding import load_embedding
# from crop_face_for_facenet import crop_face,verification_face_crop


# def load_embedding_and_verify_image(image_path=None,embedding_path=None):
#     res=[]
#     if image_path is None:
#         print(f"Failed to load image")
#         return
    
#     # crop image to 160x160 
#     img=verification_face_crop(image_path) #returns verify_cropped_img path

#     claim_embedding= get_embedding(img) #get embeding for it
    
#     os.remove(img) #delete after getting embedding 

#     # Compare to known embeddings
#     known_embeddings=load_embedding(embedding_path)

#     min_dist = float('inf')
#     for embedding in known_embeddings:
#         dist = np.linalg.norm(claim_embedding - embedding)
#         print(f'distance to embedding {embedding} is {dist}')
#         if dist < min_dist:
#             min_dist = dist
    
#         if min_dist < 0.8:
#             # res.append('present')
#             return 'Present'
        

#         # else:
#         #     res.append('not present')
#     print(f'minimum distanc {min_dist}')
#     # return res

# if __name__=="__main__":
#     res=[]
#     embedding_path='embeddings.npy'
#     img_path='cropped_test' #it could be uncropped image
#     # for folder in os.listdir(img_path):
# #     person_folder=os.path.join('testdataset', folder)
#     for img in os.listdir(img_path):
#         image=os.path.join(img_path,img)
#         # if str(folder) not in res:
#         #     res[str(folder)]=[]
#         result=load_embedding_and_verify_image(image,embedding_path)
#         res.append(result)
#         # res[str(folder)]+=result
#     print(res)
    			
    		
    		
    			
    		
import cv2
import os
import numpy as np
from get_embedding import get_embedding
from load_embedding import load_embedding
from crop_face_for_facenet import batch_crop_face,verification_face_crop


def load_embedding_and_verify_image(image_path=None, embedding_path=None):
    res = []
    if image_path is None:
        print(f"Failed to load image")
        return

    # Crop image to 160x160
    img = verification_face_crop(image_path)  # returns verify_cropped_img path
    

    claim_embedding = get_embedding(img)  # get embedding for it

    os.remove(img)  # delete after getting embedding

    # Compare to known embeddings
    known_embeddings = load_embedding(embedding_path)

    min_dist = float('inf')
    for embedding in known_embeddings:
        dist = np.linalg.norm(claim_embedding - embedding)
        print(f'distance to embedding {embedding} is {dist}')
        if dist < min_dist:
            min_dist = dist

        if min_dist < 0.8:
            # return 'Present' if a match is found
            return 'Present'

    print(f'minimum distance {min_dist}')
    # return 'Not Present' if no match is found
    return 'Not Present'


def capture_image_from_camera():
    # Open the first camera connected (usually the default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open camera")
        return None

    print("Press 'c' to capture an image.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Display the frame
        cv2.imshow("Camera Feed", frame)

        # Capture the image when 'c' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('c'):
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, frame)  # Save the captured image
            print(f"Image saved as {image_path}")
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

    return image_path


if __name__ == "__main__":
    res = []
    embedding_path = 'embeddings.npy'

    # Capture image from the camera
    img_path = capture_image_from_camera()  # Get the captured image path

    if img_path is not None:
        result = load_embedding_and_verify_image(img_path, embedding_path)
        res.append(result)

    print(res)


   
    

