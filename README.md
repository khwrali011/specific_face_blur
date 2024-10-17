# Face Blurring Flask Application

This Flask application allows users to upload an image, detect faces using the YOLOv8 model, extract face embeddings using the FaceNet model (InceptionResnetV1), and then upload a video where those faces will be automatically blurred if detected. The blurring method and similarity threshold for face detection can be customized.

## Features

- **Image Upload & Face Detection:**
  - Detect faces from an uploaded image using YOLOv8.
  - Extract and save face embeddings using FaceNet (InceptionResnetV1).
  - Store the embeddings for later use when processing videos.

- **Video Upload & Face Blurring:**
  - Upload a video where the system detects faces in each frame.
  - Blur faces in the video that match those from the uploaded image using cosine similarity.
  - Choose between different blur methods (Gaussian, Median, Bilateral).
  - Set a similarity threshold for face matching (default: 0.5).

- **Data Management:**
  - Clear all session data and delete uploaded files via the UI.

## Installation

### Prerequisites

- Python 3.x
- Pip
- [YOLOv8 model](https://github.com/ultralytics/yolov5) (Ensure the face detection model is installed: `yolov8n-face.pt`)
- Required libraries: Install the dependencies via `requirements.txt`.

### Clone the repository

```bash
git clone https://github.com/your-repo/face-blur-app.git
cd face-blur-app
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Download the YOLOv8 Face Model

Download the YOLOv8 face detection model (`yolov8n-face.pt`) and place it in the root of your project folder.

### Run the Flask app

```bash
python app.py
```

The app will be accessible on `http://localhost:8080`.

## Usage

1. **Upload Image for Face Detection:**
   - Visit the home page (`/`).
   - Upload an image containing faces.
   - The app will detect and extract the faces, then save their embeddings for later use.

2. **Upload Video for Face Blurring:**
   - After uploading an image, upload a video.
   - Set a similarity threshold (default: 0.5) to match the faces from the image and select a blur method (Gaussian, Median, or Bilateral).
   - The app will process the video and blur any detected matching faces.

3. **Download Processed Video:**
   - Once the video is processed, download it from the provided link.

4. **Clear Data:**
   - Use the "Clear Data" button to remove all uploaded files and embeddings.

## Project Structure

```bash
.
├── app.py                     # Main Flask application
├── requirements.txt            # Required Python libraries
├── templates
│   ├── index.html              # Upload interface for image and video
│   ├── download.html           # Download page for the processed video
├── static
│   └── extracted_faces         # Directory for saving extracted faces from the uploaded image
├── uploads                     # Directory for saving uploaded images and videos
└── embeddings                  # Directory for storing face embeddings in JSON format
```
