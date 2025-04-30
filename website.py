import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import io
import cv2
import tempfile
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

# Constants
FRAME_SKIP = 10  # Process every 10th frame
VIDEO_FPS = 20
VIDEO_SIZE = (640, 480)
MAX_FRAMES = 300  # Limit analysis to first 300 frames

# AQ Test Questions
AQ_QUESTIONS = [
    "I prefer to do things with others rather than on my own.",
    "I prefer to do things the same way over and over again.",
    "If I try to imagine something, I find it very easy to create a picture in my mind.",
    "I frequently get so strongly absorbed in one thing that I lose sight of other things.",
    "I often notice small sounds when others do not.",
    "I usually notice car number plates or similar strings of information.",
    "Other people frequently tell me that what I've said is impolite, even though I think it is polite.",
    "When I'm reading a story, I can easily imagine what the characters might look like.",
    "I am fascinated by dates.",
    "In a social group, I can easily keep track of several different people's conversations."
]

# Set page config
st.set_page_config(
    page_title="Multimodal Autism Detection",
    page_icon="🧠",
    layout="wide"
)

# Add title and description
st.title("Multimodal Autism Detection System")
st.markdown("""
This application uses a combination of AQ test analysis and behavioral video analysis to detect autism spectrum disorder.
Please complete the AQ test and/or upload a video for comprehensive analysis.
""")

@st.cache_resource
def load_models() -> tuple:
    """Load and cache models."""
    try:
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        with open('best_model.pkl', 'rb') as f:
            best_model = pickle.load(f)
        with open('autism_behavior_detector.pkl', 'rb') as f:
            behavior_detector = pickle.load(f)
        return encoders, best_model, behavior_detector
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

class VideoAnalyzer:
    def __init__(self):
        self.cap = None
        self.out = None
        self.temp_dir = None
        self.output_path = None
        
    def start_recording(self) -> None:
        """Initialize video recording."""
        self.temp_dir = tempfile.mkdtemp()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = os.path.join(self.temp_dir, f"webcam_{timestamp}.mp4")
        self.out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            VIDEO_FPS,
            VIDEO_SIZE
        )
        
    def stop_recording(self) -> None:
        """Clean up video resources."""
        if self.out:
            self.out.release()
        if self.cap:
            self.cap.release()
        if self.temp_dir and os.path.exists(self.temp_dir):
            if self.output_path and os.path.exists(self.output_path):
                os.unlink(self.output_path)
            os.rmdir(self.temp_dir)
            
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single video frame."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def analyze_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyze a single frame for behaviors."""
        # Placeholder for actual behavior detection
        return {
            "score": 0.75,
            "behaviors": ["Repetitive Motion", "Limited Eye Contact"]
        }

class AQTestAnalyzer:
    def __init__(self, model, encoders, feature_names):
        self.model = model
        self.encoders = encoders
        self.feature_names = feature_names
        
    def prepare_input(self, answers: List[int], user_info: Dict[str, Any]) -> pd.DataFrame:
        """Prepare input data for model prediction."""
        input_data = {}
        for feature in self.feature_names:
            if feature.startswith('A') and feature.endswith('_Score'):
                idx = int(feature[1:].split('_')[0]) - 1
                input_data[feature] = answers[idx]
            elif feature in user_info:
                input_data[feature] = user_info[feature]
        
        input_df = pd.DataFrame([input_data], columns=self.feature_names)
        
        # Apply encoders
        for column in ['gender', 'ethnicity', 'jaundice', 'contry_of_res', 'relation']:
            if column in self.encoders:
                input_df[column] = self.encoders[column].transform(input_df[column])
        
        return input_df.astype(float)
    
    def predict(self, input_df: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction using the model."""
        prediction = self.model.predict(input_df)[0]
        probability = self.model.predict_proba(input_df)[0][1]
        return {
            "prediction": prediction,
            "probability": probability
        }

def main():
    # Load models
    encoders, best_model, behavior_detector = load_models()
    if None in (encoders, best_model, behavior_detector):
        st.error("Failed to load required models. Please check the model files.")
        return
    
    # Initialize analyzers
    video_analyzer = VideoAnalyzer()
    aq_analyzer = AQTestAnalyzer(best_model, encoders, best_model.feature_names_in_)
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["AQ Test", "Video Analysis", "Combined Analysis"])
    
    with tab1:
        st.header("AQ Test Analysis")
        st.markdown("""
        Please answer the following questions on a scale of 1-4:
        - 1: Definitely agree
        - 2: Slightly agree
        - 3: Slightly disagree
        - 4: Definitely disagree
        """)
        
        # Create a form for AQ test
        with st.form("aq_test_form"):
            # User information
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=1, max_value=100, value=18)
                gender = st.selectbox("Gender", ["Male", "Female"])
                ethnicity = st.selectbox("Ethnicity", [
                    "Others", "White-European", "Middle Eastern", "Pasifika", "Black", 
                    "Hispanic", "Asian", "Turkish", "South Asian", "Latino"
                ])
            with col2:
                jaundice = st.radio("Born with Jaundice?", ["yes", "no"])
                country_of_residence = st.selectbox("Country of Residence", [
                    "Austria", "India", "United States", "South Africa", "Jordan",
                    "United Kingdom", "Brazil", "New Zealand", "Canada", "Kazakhstan",
                    "United Arab Emirates", "Australia", "Ukraine", "Iraq", "France",
                    "Malaysia", "Vietnam", "Egypt", "Netherlands", "Afghanistan",
                    "Oman", "Italy", "Bahamas", "Saudi Arabia", "Ireland", "Aruba",
                    "Sri Lanka", "Russia", "Bolivia", "Azerbaijan", "Armenia", "Serbia",
                    "Ethiopia", "Sweden", "Iceland", "Hong Kong", "Angola", "China",
                    "Germany", "Spain", "Tonga", "Pakistan", "Iran", "Argentina",
                    "Japan", "Mexico", "Nicaragua", "Sierra Leone", "Czech Republic",
                    "Niger", "Romania", "Cyprus", "Belgium", "Burundi", "Bangladesh"
                ])
                relation = st.selectbox("Relation", ["Self", "Others"])
            
            # AQ Test Questions
            st.markdown("### AQ Test Questions")
            answers = []
            for i, question in enumerate(AQ_QUESTIONS):
                answer = st.radio(
                    f"Question {i+1}: {question}",
                    options=[
                        "Definitely disagree (0)",
                        "Disagree (0.5)",
                        "Agree (1)",
                        "Definitely agree (2)"
                    ],
                    horizontal=True,
                    key=f"q_{i}"
                )
                # Map answer to score
                if "Definitely disagree" in answer:
                    score = 0
                elif "Disagree" in answer:
                    score = 0.5
                elif "Agree" in answer:
                    score = 1
                else:  # Definitely agree
                    score = 2
                answers.append(score)
            
            # Add submit button
            submitted = st.form_submit_button("Submit AQ Test")
            
            if submitted:
                with st.spinner("Analyzing AQ Test..."):
                    try:
                        # Map gender to correct format
                        gender_map = {"Male": "m", "Female": "f"}
                        mapped_gender = gender_map[gender]
                        
                        # Prepare user info
                        user_info = {
                            'age': age,
                            'gender': mapped_gender,
                            'ethnicity': ethnicity,
                            'jaundice': jaundice,
                            'contry_of_res': country_of_residence,
                            'relation': relation,
                            'result': sum(answers)
                        }
                        
                        # Prepare and make prediction
                        input_df = aq_analyzer.prepare_input(answers, user_info)
                        result = aq_analyzer.predict(input_df)
                        
                        st.success("AQ Test Analysis Complete!")
                        
                        # Display results
                        st.subheader("AQ Test Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric(label="AQ Score", value=f"{sum(answers)}")
                        with col2:
                            st.metric(label="Probability of ASD", value=f"{result['probability']*100:.1f}%")
                        
                        st.markdown("### Interpretation")
                        if result['prediction'] == 1:
                            st.warning("The AQ test results suggest a higher likelihood of autism spectrum traits.")
                        else:
                            st.success("The AQ test results suggest a lower likelihood of autism spectrum traits.")
                        
                        # Store results for combined analysis
                        st.session_state.aq_score = result['probability']
                        
                    except Exception as e:
                        st.error(f"An error occurred during AQ test analysis: {str(e)}")
    
    with tab2:
        st.header("Video Analysis")
        analysis_mode = st.radio(
            "Choose analysis mode:",
            ["Upload Video File", "Live Webcam Analysis"],
            horizontal=True
        )
        
        if analysis_mode == "Upload Video File":
            uploaded_file = st.file_uploader("Upload a video file", type=['mp4'])
            if uploaded_file and st.button("Analyze Video"):
                with st.spinner("Analyzing video..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            video_path = tmp_file.name
                        
                        cap = cv2.VideoCapture(video_path)
                        frame_count = 0
                        behaviors = []
                        
                        while cap.isOpened() and frame_count < MAX_FRAMES:
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            if frame_count % FRAME_SKIP == 0:
                                result = video_analyzer.analyze_frame(frame)
                                behaviors.extend(result["behaviors"])
                            
                            frame_count += 1
                        
                        cap.release()
                        os.unlink(video_path)
                        
                        st.success("Video Analysis Complete!")
                        display_results(behaviors, result["score"])
                        
                    except Exception as e:
                        st.error(f"Error during video analysis: {str(e)}")
        
        else:  # Live Webcam Analysis
            if 'webcam_active' not in st.session_state:
                st.session_state.webcam_active = False
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Start Webcam", disabled=st.session_state.webcam_active):
                    st.session_state.webcam_active = True
            with col2:
                if st.button("Stop Webcam", disabled=not st.session_state.webcam_active):
                    st.session_state.webcam_active = False
            
            if st.session_state.webcam_active:
                video_placeholder = st.empty()
                results_placeholder = st.empty()
                
                try:
                    video_analyzer.cap = cv2.VideoCapture(0)
                    video_analyzer.start_recording()
                    
                    frame_count = 0
                    behaviors = []
                    
                    while st.session_state.webcam_active and frame_count < MAX_FRAMES:
                        ret, frame = video_analyzer.cap.read()
                        if not ret:
                            break
                        
                        processed_frame = video_analyzer.process_frame(frame)
                        video_analyzer.out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                        video_placeholder.image(processed_frame, channels="RGB", use_column_width=True)
                        
                        if frame_count % FRAME_SKIP == 0:
                            result = video_analyzer.analyze_frame(frame)
                            behaviors.extend(result["behaviors"])
                            
                            with results_placeholder.container():
                                display_results(behaviors, result["score"])
                        
                        frame_count += 1
                    
                    video_analyzer.stop_recording()
                    st.session_state.video_score = result["score"]
                    
                except Exception as e:
                    st.error(f"Error during webcam analysis: {str(e)}")
                    st.session_state.webcam_active = False
                    video_analyzer.stop_recording()
    
    with tab3:
        st.header("Combined Analysis")
        if 'aq_score' in st.session_state and 'video_score' in st.session_state:
            combined_score = (st.session_state.aq_score * 0.4 + st.session_state.video_score * 0.6)
            display_combined_results(combined_score)
        else:
            st.info("Please complete both AQ test and video analysis to see combined results.")

def display_results(behaviors: List[str], score: float) -> None:
    """Display analysis results."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Behavior Score", value=f"{score*100:.1f}%")
    with col2:
        st.metric(label="Detected Behaviors", value=len(behaviors))
    
    if behaviors:
        st.markdown("### Detected Behaviors")
        for behavior in behaviors:
            st.write(f"- {behavior}")

def display_combined_results(score: float) -> None:
    """Display combined analysis results."""
    st.metric(label="Overall Assessment Score", value=f"{score*100:.1f}%")
    
    if score > 0.7:
        st.warning("Strong recommendation for professional evaluation")
    elif score > 0.5:
        st.info("Consider professional evaluation")
    else:
        st.success("No immediate concerns detected")

if __name__ == "__main__":
    main() 