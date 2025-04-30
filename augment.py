import os
import cv2
import numpy as np
import random
from tqdm import tqdm

class VideoAugmentor:
    def __init__(self, input_dir, output_dir, target_count=500):  # Changed to 500
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.target_count = target_count
        self.augmentation_methods = [
            'original',
            'flip_horizontal',
            'flip_vertical',
            'rotate',
            'brightness',
            'contrast',
            'blur',
            'noise',
            'crop',
            'zoom',
            'speed_change',
            'translate',
            'color_shift',
            'perspective_transform'
        ]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def augment_video(self, video_path, output_path, method):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Apply augmentation
                if method == 'original':
                    augmented_frame = frame
                elif method == 'flip_horizontal':
                    augmented_frame = cv2.flip(frame, 1)
                elif method == 'flip_vertical':
                    augmented_frame = cv2.flip(frame, 0)
                elif method == 'rotate':
                    angle = random.uniform(-15, 15)
                    M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
                    augmented_frame = cv2.warpAffine(frame, M, (width, height))
                elif method == 'brightness':
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    h, s, v = cv2.split(hsv)
                    v = cv2.add(v, random.randint(-50, 50))
                    v[v > 255] = 255
                    v[v < 0] = 0
                    final_hsv = cv2.merge((h, s, v))
                    augmented_frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
                elif method == 'contrast':
                    alpha = random.uniform(0.8, 1.2)
                    augmented_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
                elif method == 'blur':
                    k = random.choice([3, 5, 7])  # Added more blur options
                    augmented_frame = cv2.GaussianBlur(frame, (k, k), 0)
                elif method == 'noise':
                    noise = np.random.normal(0, random.randint(15, 40), frame.shape).astype(np.uint8)
                    augmented_frame = cv2.add(frame, noise)
                elif method == 'crop':
                    crop_percent = random.uniform(0.7, 0.95)  # More aggressive cropping
                    new_width = int(width * crop_percent)
                    new_height = int(height * crop_percent)
                    x = random.randint(0, width - new_width)
                    y = random.randint(0, height - new_height)
                    augmented_frame = frame[y:y+new_height, x:x+new_width]
                    augmented_frame = cv2.resize(augmented_frame, (width, height))
                elif method == 'zoom':
                    zoom_factor = random.uniform(1.0, 1.3)  # More zoom options
                    new_width = int(width / zoom_factor)
                    new_height = int(height / zoom_factor)
                    x = int((width - new_width) / 2)
                    y = int((height - new_height) / 2)
                    cropped = frame[y:y+new_height, x:x+new_width]
                    augmented_frame = cv2.resize(cropped, (width, height))
                elif method == 'translate':
                    tx = random.uniform(-0.15, 0.15) * width  # More translation
                    ty = random.uniform(-0.15, 0.15) * height
                    M = np.float32([[1, 0, tx], [0, 1, ty]])
                    augmented_frame = cv2.warpAffine(frame, M, (width, height))
                elif method == 'color_shift':
                    augmented_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    augmented_frame[:, :, 0] = (augmented_frame[:, :, 0] + random.randint(-20, 20)) % 180
                    augmented_frame = cv2.cvtColor(augmented_frame, cv2.COLOR_HSV2BGR)
                elif method == 'perspective_transform':
                    pts1 = np.float32([[0,0], [width,0], [0,height], [width,height]])
                    pts2 = np.float32([
                        [random.randint(-50, 50), random.randint(-50, 50)],
                        [width-random.randint(-50, 50), random.randint(-50, 50)],
                        [random.randint(-50, 50), height-random.randint(-50, 50)],
                        [width-random.randint(-50, 50), height-random.randint(-50, 50)]
                    ])
                    M = cv2.getPerspectiveTransform(pts1, pts2)
                    augmented_frame = cv2.warpPerspective(frame, M, (width, height))
                elif method == 'speed_change':
                    pass
                
                if method != 'speed_change':
                    out.write(augmented_frame)
            
            cap.release()
            out.release()
            
            if method == 'speed_change':
                self._change_speed(video_path, output_path, fps)
                
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals():
                out.release()
            if os.path.exists(output_path):
                os.remove(output_path)
            raise
    
    def _change_speed(self, input_path, output_path, original_fps):
        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {input_path}")
                
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            speed_factor = random.choice([0.5, 0.75, 1.25, 1.5, 1.75])  # More speed options
            new_fps = original_fps * speed_factor
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, new_fps, (width, height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            
            cap.release()
            out.release()
            
        except Exception as e:
            print(f"Error changing speed for {input_path}: {str(e)}")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            if 'out' in locals():
                out.release()
            if os.path.exists(output_path):
                os.remove(output_path)
            raise
    
    def augment_dataset(self):
        classes = [d for d in os.listdir(self.input_dir) 
                  if os.path.isdir(os.path.join(self.input_dir, d))]
        
        if not classes:
            raise ValueError(f"No class folders found in {self.input_dir}")
        
        print(f"Found {len(classes)} classes: {', '.join(classes)}")
        
        for class_name in classes:
            class_input_dir = os.path.join(self.input_dir, class_name)
            class_output_dir = os.path.join(self.output_dir, class_name)
            
            if not os.path.exists(class_input_dir):
                print(f"Warning: Input directory not found - {class_input_dir}")
                continue
                
            os.makedirs(class_output_dir, exist_ok=True)
            
            try:
                video_files = [f for f in os.listdir(class_input_dir) 
                             if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
                
                if not video_files:
                    print(f"No video files found in {class_input_dir}")
                    continue
                
                # Calculate how many augmented versions we need per original video
                per_video_target = max(1, self.target_count // len(video_files))
                
                print(f"\nAugmenting {class_name} class...")
                print(f"Found {len(video_files)} source videos")
                print(f"Generating ~{per_video_target} augmented versions per video (target: {self.target_count})")
                
                for video_file in tqdm(video_files, desc=f"Processing {class_name}"):
                    video_path = os.path.join(class_input_dir, video_file)
                    base_name = os.path.splitext(video_file)[0]
                    
                    # Generate augmented versions
                    for i in range(per_video_target):
                        method = random.choice(self.augmentation_methods)
                        output_name = f"{base_name}_aug_{i}_{method}.mp4"
                        output_path = os.path.join(class_output_dir, output_name)
                        
                        if os.path.exists(output_path):
                            continue
                            
                        self.augment_video(video_path, output_path, method)
                        
            except Exception as e:
                print(f"Error processing class {class_name}: {str(e)}")
                continue

def main():
    # Input directory - contains your original videos
    input_directory = r"C:\MY FOLDER\experiment\vedio detection\data augmentation\original_videos"
    
    # Output directory - where augmented videos will be saved
    output_directory = r"C:\MY FOLDER\experiment\vedio detection\data augmentation\augmented_videos_500"
    
    print("Video Data Augmentation Tool")
    print(f"Input directory: {input_directory}")
    print(f"Output directory: {output_directory}")
    print(f"Target count: 500 videos per class")
    
    if not os.path.exists(input_directory):
        print(f"Error: Input directory does not exist: {input_directory}")
        return
    
    try:
        augmentor = VideoAugmentor(input_directory, output_directory, target_count=500)
        augmentor.augment_dataset()
        print("\nAugmentation completed successfully!")
    except Exception as e:
        print(f"\nError during augmentation: {str(e)}")

if __name__ == "__main__":
    main()