
import cv2
import os
import numpy as np
from get_embedding import get_embedding
from load_embedding import load_embedding


def load_embedding_and_verify_image(image_path,embedding_path):
    res=[]
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Assuming image is already 160x160 cropped face
    claim_embedding= get_embedding(img)
    print(claim_embedding)

    # Compare to known embeddings
    known_embeddings=load_embedding(embedding_path)

    min_dist = float('inf')
    for embedding in known_embeddings:
        dist = np.linalg.norm(claim_embedding - embedding)
        print(f'distance to embedding {embedding} is {dist}')
        if dist < min_dist:
            min_dist = dist
    
        if min_dist < 0.8:
            res.append('present')
        

        else:
            res.append('not present')
    print(f'minimum distanc {min_dist}')
    return res

if __name__=="__main__":
    res=dict()	
    embedding_path='embeddings.npy'

    for folder in os.listdir('testdataset'):
        person_folder=os.path.join('testdataset', folder)
        for img in os.listdir(person_folder):
            img_path=os.path.join(person_folder,img)
            if str(folder) not in res:
                res[str(folder)]=[]
            result=load_embedding_and_verify_image(img_path,embedding_path)
            res[str(folder)]+=result
    print(res)
    			
    		
    		
    			
    		
  

   
    

