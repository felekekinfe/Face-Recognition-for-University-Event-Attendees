import tensorflow.compat.v1 as tf
import cv2
import numpy as np

tf.disable_v2_behavior()

# Load FaceNet model
with tf.gfile.GFile('model/20180402-114759.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def, name='')
    session = tf.compat.v1.Session(graph=graph)

images_input = graph.get_tensor_by_name("input:0")
embeddings_output = graph.get_tensor_by_name("embeddings:0")
phase_train = graph.get_tensor_by_name("phase_train:0")

"""this function used for creating a new embedding for a given picture 
which used to compare with already known embedding for verification"""
def get_embedding(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img / 255.0 - 0.5) * 2.0  
    img = np.expand_dims(img, axis=0)
    embedding = session.run(embeddings_output, feed_dict={
        images_input: img,
        phase_train: False
    })
    print(f"Embedding: {embedding[0][:5]}...")  
    return embedding[0]