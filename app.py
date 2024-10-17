import os
import json
import uuid
from flask import Flask, request, send_file, render_template, redirect, url_for, flash, session
import cv2 # type: ignore
import numpy as np
from ultralytics import YOLO # type: ignore
from facenet_pytorch import InceptionResnetV1 # type: ignore
from sklearn.metrics.pairwise import cosine_similarity # type: ignore
from PIL import Image # type: ignore
import torch # type: ignore
import logging

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Replace with a real secret key

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Initialize YOLOv8 face model
try:
    model = YOLO('yolov8n-face.pt')
except Exception as e:
    logging.error(f"Error initializing YOLO model: {e}")
    exit(1)

# Initialize FaceNet model
device = torch.device('cpu')  # Use CPU to avoid CUDA issues
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Define directories
UPLOAD_FOLDER = 'uploads'
EXTRACTED_FOLDER = 'static/extracted_faces'
EMBEDDING_FOLDER = 'embeddings'

# Ensure directories exist
for folder in [UPLOAD_FOLDER, EXTRACTED_FOLDER, EMBEDDING_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EXTRACTED_FOLDER'] = EXTRACTED_FOLDER
app.config['EMBEDDING_FOLDER'] = EMBEDDING_FOLDER


def get_face_embedding(face_image):
    try:
        face_image = Image.fromarray(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))
        face_image = face_image.resize((160, 160))
        face_tensor = torch.from_numpy(np.array(face_image)).permute(2, 0, 1).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            face_embedding = resnet(face_tensor).cpu().numpy()
        
        return face_embedding.flatten()  # Flatten the embedding to 1D
    except Exception as e:
        logging.error(f"Error calculating face embedding: {e}")
        return None

def extract_and_save_faces(image_path):
    image = cv2.imread(image_path)
    results = model.predict(source=image, conf=0.25, imgsz=1280)

    face_embeddings = []
    extracted_faces = []  # List to store paths of extracted faces
    
    if len(results) > 0:
        result = results[0]  # Access first result (assuming single image)
        boxes = result.boxes.xyxy
        
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
            face = image[y1:y2, x1:x2]
            
            # Create face embedding using VGGFace2 or facenet_pytorch
            embedding = get_face_embedding(face)
            if embedding is not None:
                face_embeddings.append(embedding.tolist())  # Convert to list

            # Save face for visualization (optional)
            face_filename = f"face_{j+1}.jpg"
            face_path = os.path.join(app.config['EXTRACTED_FOLDER'], face_filename)
            cv2.imwrite(face_path, face)
            
            # Add path to extracted_faces list
            extracted_faces.append(face_filename)
    
    return face_embeddings, extracted_faces


def process_video(video_path, face_embeddings, output_path, threshold, blur_method):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    face_embeddings_array = np.array(face_embeddings)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces in the frame
        results = model.predict(source=frame, conf=0.25, imgsz=1280)
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes.xyxy
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                face = frame[y1:y2, x1:x2]
                
                # Create face embedding
                embedding = get_face_embedding(face)
                
                if embedding is not None:
                    # Check if this face matches any of the target faces using cosine similarity
                    similarities = cosine_similarity(embedding.reshape(1, -1), face_embeddings_array)
                    if np.max(similarities) > threshold:
                        # Apply the selected blur method
                        if blur_method == 'gaussian':
                            blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
                        elif blur_method == 'median':
                            blurred_face = cv2.medianBlur(face, 99)
                        elif blur_method == 'bilateral':
                            blurred_face = cv2.bilateralFilter(face, d=9, sigmaColor=75, sigmaSpace=75)
                        
                        frame[y1:y2, x1:x2] = blurred_face
        
        # Write the frame
        out.write(frame)
    
    # Release everything
    cap.release()
    out.release()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        flash('No image file uploaded')
        return redirect(url_for('index'))
    
    file = request.files['image']
    if file.filename == '':
        flash('No image file selected')
        return redirect(url_for('index'))
    
    if file:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'target_image.jpg')
        file.save(image_path)

        # Extract faces from the uploaded image
        face_embeddings, extracted_faces = extract_and_save_faces(image_path)
        
        # Generate a unique ID for this set of embeddings
        embedding_id = str(uuid.uuid4())
        
        # Save embeddings to a file
        embedding_path = os.path.join(app.config['EMBEDDING_FOLDER'], f'{embedding_id}.json')
        with open(embedding_path, 'w') as f:
            json.dump(face_embeddings, f)
        
        # Store only the embedding ID in the session
        session['embedding_id'] = embedding_id
        
        flash(f'{len(face_embeddings)} faces detected in the image')
        
        # Pass the extracted face paths to the template
        return render_template('index.html', extracted_faces=extracted_faces)


@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'embedding_id' not in session:
        flash('Please upload an image with faces first')
        return redirect(url_for('index'))

    if 'video' not in request.files:
        flash('No video file uploaded')
        return redirect(url_for('index'))
    
    video = request.files['video']
    if video.filename == '':
        flash('No video file selected')
        return redirect(url_for('index'))
    
    if video:
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'input_video.mp4')
        video.save(video_path)
        
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_video.mp4')
        
        try:
            # Load embeddings from file
            embedding_path = os.path.join(app.config['EMBEDDING_FOLDER'], f"{session['embedding_id']}.json")
            with open(embedding_path, 'r') as f:
                face_embeddings = json.load(f)
            
            # Get the threshold and blur method from the form
            threshold = float(request.form.get('threshold', 0.5))  # Default to 0.5 if no value is selected
            blur_method = request.form.get('blur_method', 'gaussian')  # Default to Gaussian Blur
            
            process_video(video_path, face_embeddings, output_path, threshold, blur_method)
            session['output_video'] = 'output_video.mp4'
            flash('Video processed successfully')
            return redirect(url_for('show_download'))
        except Exception as e:
            logging.error(f"Error processing video: {str(e)}")
            flash(f'Error processing video: {str(e)}')
            return redirect(url_for('index'))


@app.route('/show_download')
def show_download():
    if 'output_video' not in session:
        flash('No processed video available')
        return redirect(url_for('index'))
    return render_template('download.html')

@app.route('/download')
def download_video():
    if 'output_video' not in session:
        flash('No processed video available')
        return redirect(url_for('index'))
    
    output_video_path = os.path.join(app.config['UPLOAD_FOLDER'], session['output_video'])
    return send_file(output_video_path, as_attachment=True)

@app.route('/clear', methods=['POST'])
def clear():
    # Clear session
    embedding_id = session.pop('embedding_id', None)
    session.clear()
    
    # Remove all files in upload and extracted folders
    for folder in [UPLOAD_FOLDER, EXTRACTED_FOLDER]:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logging.error(f'Failed to delete {file_path}. Reason: {e}')
    
    # Remove embedding file if it exists
    if embedding_id:
        embedding_path = os.path.join(app.config['EMBEDDING_FOLDER'], f'{embedding_id}.json')
        try:
            os.unlink(embedding_path)
        except Exception as e:
            logging.error(f'Failed to delete {embedding_path}. Reason: {e}')
    
    flash('All data cleared')
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True, port=8080)