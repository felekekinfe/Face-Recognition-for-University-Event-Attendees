import os
import numpy as np

def load_embedding(saved_path):
    if os.path.exists(saved_path):
        embedding=np.load(saved_path,allow_pickle=True).tolist()
        print(f'loaded embedding from {saved_path}')
        print(embedding)
        return embedding
    else:
        print(f'not saved embedding at {saved_path}')
print(load_embedding('embeddings.npy'))