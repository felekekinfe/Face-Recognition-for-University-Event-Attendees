
import cv2
import os
import numpy as np
from get_embedding import get_embedding




def load_image(file_path):
    img = cv2.imread(file_path)
    if img is None:
        raise ValueError(f"Could not load image at {file_path}")
    return img



def create_embedding(saving_path,folder_path):
    embeddings = []
    for person in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person)
        print(f"Person path: {person_path}")
        if os.path.isdir(person_path):
            person_embedding = []
            for photo in os.listdir(person_path):
                photo_path = os.path.join(person_path, photo)
                print(f"Photo path: {photo_path}")
                try:
                    img = load_image(photo_path)
                    embedding = get_embedding(img)
                    person_embedding.append(embedding)
                except Exception as e:
                    print(f"Failed to process {photo_path}: {e}")
            if person_embedding:
                avg_embedding = np.mean(person_embedding, axis=0)
                embeddings.append(avg_embedding)
                print(f"Embedding created for {embeddings}")
    np.save(saving_path,embeddings)
    return embeddings

if __name__=='__main__':
    saving_path='embeddings'
    folder_path='Tdataset'
    create_embedding(saving_path=saving_path,folder_path=folder_path)