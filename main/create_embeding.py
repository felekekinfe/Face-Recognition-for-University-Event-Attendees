import cv2
import os
import numpy as np
from get_embedding import get_embedding

# Function to load the image, raise error if it can't be loaded
def load_image(file_path):
    img = cv2.imread(file_path)
    print(img)
    if img is None:
        raise ValueError(f"Could not load image at {file_path}")
    return img

# Create embeddings for each person in the folder
def create_embedding(saving_path, folder_path):
    embeddings = []
    
    # Create the embeddings directory if it doesn't exist
    os.makedirs(os.path.dirname(saving_path), exist_ok=True)

    # Iterate over people in the dataset
    for person in os.listdir(folder_path):
        person_path = os.path.join(folder_path, person)
        print(f"Person path: {person_path}")

        if os.path.isdir(person_path):
            person_embedding = []
            
            # Iterate over photos of the person
            for photo in os.listdir(person_path):
                photo_path = os.path.join(person_path, photo)
                print(f"Photo path: {photo_path}")
                

                try:
                    
                    embedding = get_embedding(photo_path)  
                    person_embedding.append(embedding)
                except Exception as e:
                    print(f"Failed to process {photo_path}: {e}")
            
            # If there are embeddings for the person, calculate the average
            if person_embedding:
                avg_embedding = np.mean(person_embedding, axis=0)
                embeddings.append(avg_embedding)
                print(f"Embedding created for {person}: {avg_embedding}")
    
    # Save the embeddings to the file
    np.save(saving_path, embeddings)
    print(f"Embeddings saved to {saving_path}")
    return embeddings

if __name__ == '__main__':
    saving_path = 'embeddings/embeddings.npy'  # Ensure the extension is .npy
    folder_path = 'dataset'
    
    # Create embeddings
    create_embedding(saving_path=saving_path, folder_path=folder_path)
