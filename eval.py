import os
import pickle
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from tensorflow.keras.utils import to_categorical

# Load model and encoder
with open("autism_behavior_detector.pkl", "rb") as f:
    data = pickle.load(f)

model = data['model']
label_encoder = data['label_encoder']
input_shape = data['input_shape']
classes = data['classes']

# Constants
SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, _ = input_shape
TEST_DIR = "./train"


def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    count = 0

    while success and count < SEQUENCE_LENGTH:
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        frame = frame / 255.0
        frames.append(frame)
        success, frame = cap.read()
        count += 1

    # Pad if needed
    while len(frames) < SEQUENCE_LENGTH:
        frames.append(np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    
    cap.release()
    return np.array(frames)

# Load test videos
X_test = []
y_test = []

for label in classes:
    class_dir = os.path.join(TEST_DIR, label)
    if not os.path.exists(class_dir):
        print(f"Missing directory: {class_dir}")
        continue

    for video_file in os.listdir(class_dir):
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            continue

        video_path = os.path.join(class_dir, video_file)
        try:
            frames = extract_frames(video_path)
            X_test.append(frames)
            y_test.append(label)
        except Exception as e:
            print(f"Error with {video_file}: {str(e)}")

# Check if any test data was found
if not X_test:
    raise ValueError("No test data found!")

# Prepare inputs
X_test = np.array(X_test)
y_test_encoded = label_encoder.transform(y_test)
y_test_categorical = to_categorical(y_test_encoded, num_classes=len(classes))

# Predict
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"\n✅ Model Accuracy on Test Set: {accuracy * 100:.2f}%")

