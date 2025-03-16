


# Face-Recognition-for-University-Event-Attendees

## Overview
**Face Recognition for the 3rd-Year Half-Life Celebration** – A computer vision-powered face recognition system designed to verify attendance at the 3rd-year "Half-Life Celebration" event of the Computer Science and Engineering (CSE) department. This project captures a face via webcam, generates an embedding using FaceNet, and compares it against precomputed embeddings of event attendees stored in a `.npy` file. If a match is found (based on KNN distance), it displays "Present" in green; otherwise, it displays "Not Present" in red on the camera feed for 3 seconds.

### Key Features
- **Live Face Detection**: Detects faces in real-time using OpenCV with Haar Cascades.
- **Face Cropping**: Uses MTCNN to crop faces from group photos for consistent embedding generation.
- **Organized Dataset**: Manually assigned folders for each person to streamline embedding generation.
- **Embedding Generation**: Leverages FaceNet to create 128D face embeddings.
- **Embedding Storage**: Saves and loads embeddings using NumPy `.npy` files for efficient comparison.
- **Distance-Based Matching**: Computes Euclidean distance (KNN) between embeddings with a threshold of 0.7.
- **Event-Specific Verification**: Tailored for the 3rd-year Half-Life Celebration, a milestone event for CSE students.

## Technologies Used
- **Python 3.8+**
- **OpenCV**: For webcam access and face detection via Haar Cascades.
- **MTCNN**: For precise face detection and cropping from group photos.
- **FaceNet**: Pre-trained model for generating face embeddings.
- **TensorFlow/Keras**: For loading and running the FaceNet model.
- **NumPy**: For numerical operations, distance calculations, and `.npy` file handling.
- **Hardware**: A webcam for live face capture.

## Project Structure
Here’s the folder structure of the repository:

```
├── embeddings/                # Directory for precomputed embeddings
│   └── embeddings.npy        # Precomputed embeddings of Half-Life Celebration attendees
├── experiment.ipynb           # Jupyter notebook for experiments and testing
├── LICENSE                    # Project license file
├── main/                      # Source code directory
│   ├── create_embedding.py    # Script to generate embeddings from images
│   ├── crop_face_for_facenet.py  # Script to crop faces using MTCNN
│   ├── get_embedding.py       # Script to generate embeddings using FaceNet
│   ├── load_embedding.py      # Script to load precomputed embeddings from .npy files
│   ├── verify_from_webcam.py  # Main script for webcam-based face verification
├── model/                     # Directory for FaceNet model weights
│    ├──facenet.pb             # FaceNet model weights file
├── dataset/                   # Directory for dataset (group photos and cropped faces)
│                
│                              # Directory containing subfolders for each person
│       ├── person1/           # Cropped faces of person1
│       ├── person2/           # Cropped faces of person2
│       └── ...                # Additional folders for other attendees
├── .gitignore                 # Git ignore file
└── requirements.txt           # Python dependencies file
```

### File Descriptions
- **`embeddings/embeddings.npy`**: Contains embeddings of faces from the 3rd-year Half-Life Celebration event.
- **`main/create_embedding.py`**: Generates embeddings for a batch of images (organized by person) using FaceNet and saves them as `.npy`.
- **`main/crop_face_for_facenet.py`**: Crops faces from group photos using MTCNN, resizing them to 160x160 pixels for FaceNet.
- **`main/get_embedding.py`**: Generates an embedding for a single face image using FaceNet.
- **`main/load_embedding.py`**: Loads precomputed embeddings from a `.npy` file for comparison.
- **`main/verify_from_webcam.py`**: Main script that captures a face via webcam, generates its embedding, compares it to loaded embeddings, and displays the result.
- **`model/20180402-114759.pb`**: Pre-trained FaceNet model weights file for embedding generation.
- **`dataset/cropped_faces/`**: Directory where cropped faces are organized into subfolders for each person (e.g., `person1/`, `person2/`).
- **`requirements.txt`**: Lists all Python dependencies required for the project.

## How It Works
1. **Dataset Preparation**:
   - Group photos from the 3rd-year Half-Life Celebration were manually collected.
   - Faces were cropped from these photos using MTCNN (`main/crop_face_for_facenet.py`).
   - Cropped faces were manually organized into subfolders within `Tdataset/cropped_faces/`, with each subfolder representing one person (e.g., `person1/`, `person2/`).
   - Embeddings were generated for each person’s images using FaceNet (`main/create_embedding.py`) and saved to `embeddings/embeddings.npy`.

2. **Real-time Verification**:
   - The webcam captures a live feed (`main/verify_from_webcam.py`).
   - OpenCV with Haar Cascades detects a face in the frame.
   - The face is cropped and resized to 160x160 pixels for FaceNet compatibility.
   - An embedding is generated for the captured face using FaceNet (`main/get_embedding.py`).
   - Precomputed embeddings are loaded from `embeddings/embeddings.npy` using `main/load_embedding.py`.
   - The new embedding is compared to the loaded embeddings using Euclidean distance (KNN).
   - If the minimum distance is less than 0.8, "Present" is displayed in green; otherwise, "Not Present" is shown in red on the camera feed for 3 seconds.

## Prerequisites
Before running the project, ensure you have the following:

### Software
- Python 3.8 or higher
- `pip` for installing dependencies

### Dependencies
Install the required packages using:
```bash
pip install -r requirements.txt
```
Create a `requirements.txt` file with the following content:
```
opencv-python==4.5.5.64
mtcnn==0.1.1
tensorflow==2.10.0
numpy==1.23.5
keras-facenet==0.3.2
```

### Hardware
- A working webcam (built-in or external) for live face capture.
- (Optional) GPU for faster FaceNet inference.

### Pre-trained Model
- Download the FaceNet model weights (e.g., `20180402-114759.pb`) and place them in the `main/model/` directory. You can find this model on GitHub repositories like [david-sandberg/facenet](https://github.com/david-sandberg/facenet).

## Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/felekekinfe/Face-Recognition-for-University-Event-Attendees
   cd Face-Recognition-for-University-Event-Attendees
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**:
   - **Step 1: Collect Group Photos**:
     - Place group photos from the Half-Life Celebration event in the `Tdataset/` directory.
   - **Step 2: Crop Faces**:
     - Run the cropping script to extract faces from group photos:
       ```bash
       python main/crop_face_for_facenet.py
       ```
     - This will save cropped faces in a temporary directory (e.g., `dataset/`).
   - **Step 3: Manually Organize Faces**:
     - Manually sort the cropped faces into subfolders within `dataset/cropped_faces/`, with each subfolder named after a person (e.g., `person1/`, `person2/`).
     - Ensure each subfolder contains only the cropped faces of that specific person.
   - **Step 4: Generate Embeddings**:
     - Run the embedding generation script to process the organized faces:
       ```bash
       python main/create_embedding.py
       ```
     - This script should iterate through each person’s folder, generate embeddings for their faces, and save them to `embeddings/embeddings.npy`.

4. **Add FaceNet Model**:
   - Place the FaceNet model weights (`20180402-114759.pb`) in the `model/` directory.

## Usage
1. **Run the Verification Script**:
   ```bash
   python main/verify_from_webcam.py
   ```
   - The webcam will open, showing a live feed.
   - When a face is detected, the system will:
     - Generate an embedding for the detected face.
     - Load precomputed embeddings from `embeddings/embeddings.npy`.
     - Compare the embeddings and display "Present" (green) or "Not Present" (red) above the detected face.
   - The result will be shown for 3 seconds before closing.

2. **Optional: Test with Pre-cropped Images**:
   - Use `experiment.ipynb` to test the pipeline with pre-cropped faces or debug issues.

## Troubleshooting
- **Camera Not Opening**:
  - Ensure your webcam is connected and not in use by another application.
  - Check camera permissions in your OS settings.
  - Try changing the camera index in `main/verify_from_webcam.py` (e.g., `cv2.VideoCapture(1)`).
  - Run the following to check OpenCV video support:
    ```python
    import cv2
    print(cv2.getBuildInformation())
    ```

## License
This project is licensed under the [MIT License](LICENSE).

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for enhancements or bug fixes.

## Acknowledgments
- [FaceNet](https://arxiv.org/abs/1503.03832) for face embedding generation.
- [MTCNN](https://arxiv.org/abs/1604.02878) for face detection and cropping.
- The CSE department for organizing the memorable 3rd-year Half-Life Celebration event.
