import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM, Dense, TimeDistributed
from tensorflow.keras.utils import to_categorical
import pickle

# Configuration
SEQUENCE_LENGTH = 20  # Number of frames per video sequence
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
DATASET_DIR = "dataset"
MODEL_FILENAME = "autism_behavior_detector.pkl"
CLASSES = ["headbanging", "arm_flapping", "spinning"]

def extract_frames(video_path):
    """
    Extract frames from a video file
    """
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    
    while success and count < SEQUENCE_LENGTH:
        # Resize and normalize frame
        image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
        image = image / 255.0
        frames.append(image)
        success, image = vidcap.read()
        count += 1
    
    # Pad with black frames if video is shorter than SEQUENCE_LENGTH
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    
    return np.array(frames)

def load_dataset():
    """
    Load and process all videos in the dataset with better path handling
    """
    X = []
    y = []
    
    # Get the absolute path to the dataset directory
    base_dir = os.path.abspath("C:\MY FOLDER\experiment\\vedio detection\datasets222\Dataset")
    print(f"Looking for dataset in: {base_dir}")
    
    for class_name in CLASSES:
        class_dir = os.path.join(base_dir, "train", class_name)
        
        # Check if directory exists
        if not os.path.exists(class_dir):
            print(f"Warning: Directory not found - {class_dir}")
            continue
            
        print(f"Processing class: {class_name} in {class_dir}")
        
        video_files = [f for f in os.listdir(class_dir) 
                      if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            print(f"Warning: No video files found in {class_dir}")
            continue
            
        for video_file in video_files:
            video_path = os.path.join(class_dir, video_file)
            try:
                frames = extract_frames(video_path)
                X.append(frames)
                y.append(class_name)
                print(f"Processed: {video_file}")
            except Exception as e:
                print(f"Error processing {video_file}: {str(e)}")
                continue
    
    if not X:
        raise ValueError("No video data found in the specified directories. Please check your dataset paths.")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    
    return np.array(X), y_categorical, le 

def create_model(input_shape, num_classes):
    """
    Create a CNN-LSTM model for video classification
    """
    model = Sequential()
    
    # CNN layers to process each frame
    model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu'), 
                              input_shape=input_shape))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Flatten()))
    
    # LSTM layer to process sequences
    model.add(LSTM(64))
    
    # Dense layers for classification
    model.add(Dense(64, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def save_model(model, label_encoder):
    """
    Save the model and label encoder to a pickle file
    """
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'input_shape': (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        'classes': CLASSES
    }
    
    with open(MODEL_FILENAME, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved to {MODEL_FILENAME}")

def main():
    # Load and prepare dataset
    X, y, label_encoder = load_dataset()
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    model = create_model(input_shape, len(CLASSES))
    
    print("Training model...")
    history = model.fit(X_train, y_train,
                       validation_data=(X_val, y_val),
                       epochs=10,
                       batch_size=8)
    
    # Save the trained model
    save_model(model, label_encoder)


    
    return X_val, y_val


if __name__ == "__main__":
    main()