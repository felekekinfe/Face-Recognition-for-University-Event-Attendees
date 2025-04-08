

# Face Recognition for University Event Attendees


*Celebrating Milestones with Cutting-Edge Technology*

## 🌟 Overview
**Face Recognition for the 3rd-Year Half-Life Celebration** is a computer vision-powered system designed to verify attendance at the 3rd-year "Half-Life Celebration" event of the Computer Science and Engineering (CSE) department. This project captures faces via webcam, generates embeddings using FaceNet, and compares them against precomputed embeddings stored in a `.npy` file. With real-time feedback—displaying "Present" in green or "Not Present" in red—this tool brings efficiency and fun to event management!

### ✨ Key Features
- **📸 Live Face Detection**: Real-time face detection using OpenCV with Haar Cascades.
- **✂️ Face Cropping**: Precise face extraction from group photos with MTCNN.
- **📂 Organized Dataset**: Manually curated folders for each attendee.
- **🧠 Embedding Generation**: 128D face embeddings via FaceNet.
- **💾 Embedding Storage**: Efficient `.npy` file handling with NumPy.
- **📏 Distance-Based Matching**: KNN with Euclidean distance (threshold: 0.7).
- **🎉 Event-Specific**: Tailored for the CSE 3rd-year Half-Life Celebration.

---

## 🛠️ Technologies Used
- **🐍 Python 3.8+**
- **📷 OpenCV**: Webcam access and Haar Cascade face detection.
- **🔍 MTCNN**: Advanced face detection and cropping.
- **🤖 FaceNet**: Pre-trained model for embeddings.
- **🧮 TensorFlow/Keras**: FaceNet model execution.
- **🔢 NumPy**: Numerical operations and `.npy` management.
- **💻 Hardware**: Webcam (GPU optional for faster inference).

---

## 📁 Project Structure
```
├── embeddings/                # Precomputed embeddings
│   └── embeddings.npy        # Attendee embeddings
├── experiment.ipynb          # Testing and experimentation notebook
├── LICENSE                   # MIT License file
├── main/                     # Core scripts
│   ├── create_embedding.py   # Batch embedding generation
│   ├── crop_face_for_facenet.py  # Face cropping with MTCNN
│   ├── get_embedding.py      # Single face embedding
│   ├── load_embedding.py     # Load `.npy` embeddings
│   ├── verify_from_webcam.py # Webcam verification script
├── model/                    # FaceNet model weights
│   └── facenet.pb           # Pre-trained FaceNet weights
├── dataset/                  # Event photos and cropped faces
│   ├── person1/             # Cropped faces of person1
│   ├── person2/             # Cropped faces of person2
│   └── ...                  # More attendee folders
├── .gitignore               # Git ignore file
└── requirements.txt         # Python dependencies
```

### 📜 File Descriptions
- **`embeddings/embeddings.npy`**: Stores embeddings for Half-Life Celebration attendees.
- **`main/create_embedding.py`**: Generates and saves embeddings from organized images.
- **`main/crop_face_for_facenet.py`**: Crops faces to 160x160 pixels using MTCNN.
- **`main/get_embedding.py`**: Creates an embedding for a single face.
- **`main/load_embedding.py`**: Loads embeddings from `.npy` files.
- **`main/verify_from_webcam.py`**: Runs live verification via webcam.
- **`model/facenet.pb`**: FaceNet model weights.
- **`dataset/`**: Contains group photos and cropped faces in subfolders (e.g., `person1/`).
- **`requirements.txt`**: Lists all required Python packages.

---

## 🚀 How It Works
1. **Dataset Preparation**:
   - Collect group photos from the Half-Life Celebration.
   - Crop faces using `crop_face_for_facenet.py`.
   - Organize cropped faces into `dataset/` subfolders (e.g., `person1/`).
   - Generate embeddings with `create_embedding.py` and save to `embeddings.npy`.

2. **Real-Time Verification**:
   - Launch `verify_from_webcam.py` to start the webcam.
   - Detect faces with OpenCV Haar Cascades.
   - Crop and resize faces for FaceNet.
   - Generate an embedding and compare it to stored embeddings.
   - Display "Present" (green) or "Not Present" (red) for 3 seconds.

---

## ⚙️ Prerequisites

### Software
- **Python 3.8+**
- **`pip`** for dependency management

### Dependencies
Install via:
```bash
pip install -r requirements.txt
```
**`requirements.txt`:**
```
opencv-python==4.5.5.64
mtcnn==0.1.1
tensorflow==2.10.0
numpy==1.23.5
keras-facenet==0.3.2
```

### Hardware
- **Webcam**: Built-in or external.
- **GPU** (optional): Speeds up FaceNet inference.

### Pre-trained Model
- Download `20180402-114759.pb` from [david-sandberg/facenet](https://github.com/david-sandberg/facenet) and place it in `model/`.

---

## 🏁 Setup
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
   - **Collect Photos**: Add group photos to `dataset/`.
   - **Crop Faces**:
     ```bash
     python main/crop_face_for_facenet.py
     ```
   - **Organize Faces**: Sort cropped faces into `dataset/` subfolders (e.g., `person1/`).
   - **Generate Embeddings**:
     ```bash
     python main/create_embedding.py
     ```

4. **Add FaceNet Model**:
   - Place `facenet.pb` in `model/`.

---

## 🎬 Usage
1. **Run Verification**:
   ```bash
   python main/verify_from_webcam.py
   ```
   - Webcam opens, detects faces, and displays attendance status.

2. **Test Pipeline**:
   - Use `experiment.ipynb` for debugging or testing with pre-cropped images.

---

## 🔧 Troubleshooting
- **Camera Issues**:
  - Check webcam connection and permissions.
  - Adjust camera index in `verify_from_webcam.py` (e.g., `cv2.VideoCapture(1)`).
  - Verify OpenCV video support:
    ```python
    import cv2
    print(cv2.getBuildInformation())
    ```

---

## 📜 License
Licensed under the [MIT License](LICENSE).

---

## 🤝 Contributing
We welcome contributions! Open an issue or submit a pull request to enhance this project.

---

## 🙏 Acknowledgments
- **[FaceNet](https://arxiv.org/abs/1503.03832)**: For robust face embeddings.
- **[MTCNN](https://arxiv.org/abs/1604.02878)**: For precise face detection.
- **CSE Department**: For hosting the unforgettable Half-Life Celebration.

---

*Built with 💻 and ❤️ by [felekekinfe](https://github.com/felekekinfe).*  
Let’s make event attendance smarter, one face at a time! 🚀

---

