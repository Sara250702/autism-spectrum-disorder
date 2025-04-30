import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Set page config with custom theme
st.set_page_config(
    page_title="Autism Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .css-1d391kg {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stRadio > div {
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
    }
    .stSelectbox > div {
        background-color: #ffffff;
        border-radius: 5px;
    }
    .stNumberInput > div {
        background-color: #ffffff;
        border-radius: 5px;
    }
    .stMarkdown h1 {
        color: #2c3e50;
        text-align: center;
        margin-bottom: 30px;
    }
    .stMarkdown h2 {
        color: #34495e;
        margin-top: 20px;
    }
    .stMarkdown h3 {
        color: #7f8c8d;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model and encoders
@st.cache_resource
def load_model_and_encoders():
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    return model, encoders

# AQ Test Questions
questions = [
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

# Initialize session state for storing answers
if 'answers' not in st.session_state:
    st.session_state.answers = [0] * len(questions)

# Main app
def main():
    # Header with logo and title
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🧠 Autism Spectrum Assessment</h1>", unsafe_allow_html=True)
    
    # Landing page content
    if 'test_started' not in st.session_state:
        st.session_state.test_started = False
        
        # Hero section
        st.markdown("""
        <div style='background-color: #e3f2fd; padding: 40px; border-radius: 15px; margin-bottom: 30px;'>
            <h2 style='text-align: center; color: #1976d2;'>Understanding Autism Spectrum</h2>
            <p style='text-align: center; font-size: 18px; color: #424242;'>
                A comprehensive assessment tool to help identify autism spectrum traits
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Information cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: #1976d2;'>📊 About the Test</h3>
                <p style='color: #424242;'>
                    This assessment uses the Autism Spectrum Quotient (AQ) test, a scientifically validated tool developed by researchers at the Autism Research Centre.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: #1976d2;'>⏱️ Time Required</h3>
                <p style='color: #424242;'>
                    The test takes approximately 10-15 minutes to complete. Please ensure you have enough time to answer all questions thoughtfully.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='color: #1976d2;'>🔒 Privacy</h3>
                <p style='color: #424242;'>
                    Your responses are completely anonymous and will be used solely for assessment purposes. No personal data is stored.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Key Features
        st.markdown("""
        <div style='background-color: #f5f5f5; padding: 30px; border-radius: 15px; margin: 20px 0;'>
            <h2 style='text-align: center; color: #1976d2;'>Key Features</h2>
            <div style='display: flex; justify-content: space-around; margin-top: 20px;'>
                <div style='text-align: center;'>
                    <span style='font-size: 24px;'>🔍</span>
                    <p>Comprehensive Assessment</p>
                </div>
                <div style='text-align: center;'>
                    <span style='font-size: 24px;'>📈</span>
                    <p>Detailed Scoring</p>
                </div>
                <div style='text-align: center;'>
                    <span style='font-size: 24px;'>💡</span>
                    <p>Insightful Results</p>
                </div>
                <div style='text-align: center;'>
                    <span style='font-size: 24px;'>🛡️</span>
                    <p>Secure & Private</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Disclaimer and Start Button
        st.markdown("""
        <div style='background-color: #fff3e0; padding: 20px; border-radius: 10px; margin: 20px 0;'>
            <h3 style='color: #e65100;'>⚠️ Important Note</h3>
            <p style='color: #424242;'>
                This is a screening tool and not a diagnostic test. The results should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions you may have regarding a medical condition.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Start Test Button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Begin Assessment", use_container_width=True, type="primary"):
                st.session_state.test_started = True
                st.experimental_rerun()
        
        return
    
    # If test has started, show the test interface
    # Introduction
    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <p style='font-size: 18px; color: #34495e;'>
            Please answer the following questions honestly based on your preferences and experiences.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for user information with improved styling
    with st.sidebar:
        # Logo with gradient background
        st.markdown("""
        <div style='
            background: linear-gradient(135deg, #2196F3 0%, #E3F2FD 50%, #2196F3 100%);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        '>
            <h1 style='
                color: #1976D2;
                font-size: 24px;
                margin: 0;
                padding: 10px;
                font-weight: bold;
                text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
            '>🧠 Autism Spectrum</h1>
            <p style='
                color: #1976D2;
                font-size: 16px;
                margin: 0;
                opacity: 0.9;
                text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
            '>Detection System</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h2 style='color: #2c3e50;'>👤 User Information</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        age = st.number_input("Age", min_value=1, max_value=100, value=18)
        gender = st.selectbox("Gender", [
            "Male", "Female", "Non-binary", "Genderqueer", "Genderfluid", 
            "Agender", "Bigender", "Two-Spirit", "Prefer not to say"
        ])
        ethnicity = st.selectbox("Ethnicity", [
            "White-European", "Middle Eastern", "Pasifika", "Black", 
            "Hispanic", "Asian", "Turkish", "South Asian", "Latino",
            "Indigenous", "Mixed Race", "Arab", "Jewish", "Romani",
            "Southeast Asian", "Central Asian", "North African",
            "Sub-Saharan African", "Pacific Islander", "Prefer not to say"
        ])
        jaundice = st.radio("Born with Jaundice?", ["yes","no"])
        country_of_residence = st.selectbox("Country of Residence", [
            "United States", "United Kingdom", "Canada", "Australia", "New Zealand",
            "India", "China", "Japan", "South Korea", "Singapore",
            "Germany", "France", "Italy", "Spain", "Netherlands",
            "Brazil", "Mexico", "Argentina", "Chile", "Colombia",
            "South Africa", "Nigeria", "Kenya", "Egypt", "Morocco",
            "Russia", "Ukraine", "Poland", "Sweden", "Norway",
            "Saudi Arabia", "United Arab Emirates", "Qatar", "Turkey", "Israel",
            "Indonesia", "Malaysia", "Thailand", "Vietnam", "Philippines",
            "Other"
        ])
        relation = st.selectbox("Relation", [
            "Self", "Parent", "Guardian", "Sibling", "Spouse/Partner",
            "Friend", "Healthcare Provider", "Teacher", "Other"
        ])
    
    # Main content area with improved layout
    st.markdown("<h2 style='text-align: center;'>📝 Autism Spectrum Quotient (AQ) Test</h2>", unsafe_allow_html=True)
    
    # Instructions in a styled container
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: #2c3e50;'>Instructions</h3>
        <p style='color: #34495e;'>
            For each question, please select one of the following options:
        </p>
        <ul style='color: #34495e;'>
            <li><strong>Disagree</strong> (Score: 0)</li>
            <li><strong>Slightly Disagree</strong> (Score: 0)</li>
            <li><strong>Slightly Agree</strong> (Score: 1)</li>
            <li><strong>Agree</strong> (Score: 2)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Display questions in a more organized way
    for i, question in enumerate(questions):
        st.markdown(f"""
        <div style='background-color: #ffffff; padding: 15px; border-radius: 10px; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
            <h3 style='color: #2c3e50;'>Question {i+1}</h3>
            <p style='color: #34495e;'>{question}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Replace radio with slider
        answer = st.slider(
            f"Select your answer for Question {i+1}",
            min_value=0,
            max_value=3,
            value=0,
            step=1,
            format="%d",
            key=f"q_{i}",
            label_visibility="collapsed"
        )
        
        # Map slider value to score
        # 0: Disagree (0 points), 1: Slightly Disagree (0.5 points), 2: Slightly Agree (1 point), 3: Agree (2 points)
        if answer == 0:  # Disagree
            st.session_state.answers[i] = 0
        elif answer == 1:  # Slightly Disagree
            st.session_state.answers[i] = 0.5
        elif answer == 2:  # Slightly Agree
            st.session_state.answers[i] = 1
        else:  # Agree
            st.session_state.answers[i] = 2
            
        # Add labels below slider
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("<p style='text-align: center; color: #34495e;'>Disagree (0)</p>", unsafe_allow_html=True)
        with col2:
            st.markdown("<p style='text-align: center; color: #34495e;'>Slightly Disagree (0.5)</p>", unsafe_allow_html=True)
        with col3:
            st.markdown("<p style='text-align: center; color: #34495e;'>Slightly Agree (1)</p>", unsafe_allow_html=True)
        with col4:
            st.markdown("<p style='text-align: center; color: #34495e;'>Agree (2)</p>", unsafe_allow_html=True)
    
    # Calculate total score
    total_score = sum(st.session_state.answers)
    
    # Prediction button with improved styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Get Prediction", use_container_width=True):
            # Load model and encoders
            model, encoders = load_model_and_encoders()
            
            # Get feature names from the model
            feature_names = model.feature_names_in_
            
            # Map gender to correct format
            gender_map = {"Male": "m", "Female": "f"}
            mapped_gender = gender_map[gender]
            
            # Prepare input features
            input_data = {
                'A1_Score': st.session_state.answers[0],
                'A2_Score': st.session_state.answers[1],
                'A3_Score': st.session_state.answers[2],
                'A4_Score': st.session_state.answers[3],
                'A5_Score': st.session_state.answers[4],
                'A6_Score': st.session_state.answers[5],
                'A7_Score': st.session_state.answers[6],
                'A8_Score': st.session_state.answers[7],
                'A9_Score': st.session_state.answers[8],
                'A10_Score': st.session_state.answers[9],
                'age': age,
                'gender': mapped_gender,
                'ethnicity': ethnicity,
                'jaundice': jaundice,
                'contry_of_res': country_of_residence,
                'relation': relation,
                'result': total_score
            }
            
            # Convert to DataFrame with model's feature order
            input_df = pd.DataFrame([input_data], columns=feature_names)
            
            # Apply encoders to categorical features
            for column in ['gender', 'ethnicity', 'jaundice', 'contry_of_res', 'relation']:
                if column in encoders:
                    input_df[column] = encoders[column].transform(input_df[column])
            
            # Ensure all features are numeric
            input_df = input_df.astype(float)
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            # Display results in a styled container
            st.markdown("""
            <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h2 style='color: #2c3e50; text-align: center;'>Results</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Model accuracy information
            st.markdown("""
            <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                <p style='color: #2c3e50; text-align: center; margin: 0;'>
                    <strong>Model Accuracy:</strong> 95.2% (based on cross-validation)
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            if prediction == 1:
                st.error(f"Based on the AQ test, there is a {probability*100:.2f}% probability of autism spectrum traits.")
            else:
                st.success(f"Based on the AQ test, there is a {probability*100:.2f}% probability of autism spectrum traits.")
            
            # Display total score in a metric card
            st.markdown(f"""
            <div style='background-color: #e3f2fd; padding: 15px; border-radius: 10px; text-align: center; margin-top: 20px;'>
                <h3 style='color: #2c3e50;'>Total AQ Score</h3>
                <h2 style='color: #1976d2;'>{total_score}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Disclaimer
            st.info("""
            **Note:** This is a screening tool and not a diagnostic test. 
            If you have concerns, please consult with a healthcare professional.
            """)

if __name__ == "__main__":
    main() 