import streamlit as st
import cv2
import numpy as np
import pickle
import tempfile
import os
from PIL import Image
import time

# Constants
SEQUENCE_LENGTH = 20
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64

# Load the model
@st.cache_resource
def load_model():
    with open('autism_behavior_detector.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

def extract_frames(video_path):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    
    while success and count < SEQUENCE_LENGTH:
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = image / 255.0
        frames.append(image)
        success, image = vidcap.read()
        count += 1
    
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    
    return np.array(frames)

def process_webcam_frames():
    model_data = load_model()
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    classes = model_data['classes']
    
    st.title("Autism Behavior Detection System")
    st.write("This system detects three types of behaviors: headbanging, arm_flapping, and spinning")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Create placeholders for the video feed and predictions
    video_placeholder = st.empty()
    prediction_placeholder = st.empty()
    
    # Buffer to store frames
    frame_buffer = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break
                
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize frame for display
            display_frame = cv2.resize(frame_rgb, (640, 480))
            
            # Process frame for prediction
            processed_frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
            processed_frame = processed_frame / 255.0
            frame_buffer.append(processed_frame)
            
            # Keep only the last SEQUENCE_LENGTH frames
            if len(frame_buffer) > SEQUENCE_LENGTH:
                frame_buffer.pop(0)
            
            # Make prediction when we have enough frames
            if len(frame_buffer) == SEQUENCE_LENGTH:
                prediction = model.predict(np.array([frame_buffer]))
                predicted_class = classes[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                # Display prediction
                prediction_placeholder.markdown(f"""
                **Detected Behavior:** {predicted_class}
                **Confidence:** {confidence:.2f}%
                """)
                
                # Store results in session state
                st.session_state.video_results = {
                    'behavior': predicted_class,
                    'confidence': confidence
                }
            
            # Display video feed
            video_placeholder.image(display_frame, channels="RGB")
            
            # Add a small delay to control frame rate
            time.sleep(0.1)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        cap.release()

def process_uploaded_video(uploaded_file):
    model_data = load_model()
    model = model_data['model']
    label_encoder = model_data['label_encoder']
    classes = model_data['classes']
    
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    try:
        # Extract frames
        frames = extract_frames(video_path)
        
        # Make prediction
        prediction = model.predict(np.array([frames]))
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Store results in session state
        st.session_state.video_results = {
            'behavior': predicted_class,
            'confidence': confidence
        }
        
        # Display results
        st.success(f"""
        **Detected Behavior:** {predicted_class}
        **Confidence:** {confidence:.2f}%
        """)
        
        # Display video
        st.video(uploaded_file)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary file
        os.unlink(video_path)

def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Webcam Detection", "Upload Video"]
    )
    
    if app_mode == "Webcam Detection":
        process_webcam_frames()
    else:
        st.title("Upload Video for Analysis")
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        if uploaded_file is not None:
            process_uploaded_video(uploaded_file)

if __name__ == "__main__":
    main() 