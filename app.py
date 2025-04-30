import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Autism Detection System",
    page_icon="🧠",
    layout="wide"
)

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
    st.title("🧠 Autism Detection System")
    st.markdown("""
    This application uses the Autism Spectrum Quotient (AQ) test to help assess the likelihood of autism spectrum traits.
    Please answer the following questions honestly based on your preferences and experiences.
    """)
    
    # Sidebar for user information
    with st.sidebar:
        st.header("User Information")
        age = st.number_input("Age", min_value=1, max_value=100, value=18)
        gender = st.selectbox("Gender", ["Male", "Female"])  # Will be mapped to ["m", "f"]
        ethnicity = st.selectbox("Ethnicity", [
            "Others", "White-European", "Middle Eastern", "Pasifika", "Black", 
            "Hispanic", "Asian", "Turkish", "South Asian", "Latino"
        ])  # Will be encoded to [5, 9, 4, 6, 1, 2, 0, 8, 7, 3]
        jaundice = st.radio("Born with Jaundice?", ["yes", "no"])  # Will be encoded to [0, 1]
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
        ])  # Will be encoded to [6, 24, 53, 45, 30, 52, 12, 35, 14, 31, 51, 5, 50, 26, 20, 32, 54, 18, 34, 0, 38, 28, 8, 42, 27, 4, 47, 41, 11, 7, 3, 43, 19, 48, 23, 22, 1, 15, 21, 46, 49, 39, 25, 2, 29, 33, 36, 44, 17, 37, 40, 16, 10, 13, 9]
        relation = st.selectbox("Relation", ["Self", "Others"])  # Will be encoded to [1, 0]
    
    # Main content area
    st.header("Autism Spectrum Quotient (AQ) Test")
    st.markdown("""
    For each question, please select one of the following options:
    - **Disagree** (Score: 0)
    - **Slightly Disagree** (Score: 0)
    - **Slightly Agree** (Score: 1)
    - **Agree** (Score: 1)
    """)
    
    # Display questions and collect answers
    for i, question in enumerate(questions):
        st.subheader(f"Question {i+1}")
        st.write(question)
        answer = st.radio(
            f"Select your answer for Question {i+1}",
            ["Disagree", "Slightly Disagree", "Slightly Agree", "Agree"],
            key=f"q_{i}",
            horizontal=True
        )
        
        # Map answer to score
        if answer in ["Slightly Agree", "Agree"]:
            st.session_state.answers[i] = 1
        else:
            st.session_state.answers[i] = 0
    
    # Calculate total score
    total_score = sum(st.session_state.answers)
    
    # Prediction button
    if st.button("Get Prediction"):
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
        
        # Display results
        st.subheader("Results")
        if prediction == 1:
            st.error(f"Based on the AQ test, there is a {probability*100:.2f}% probability of autism spectrum traits.")
            st.info("""
            **Note:** This is a screening tool and not a diagnostic test. 
            If you have concerns, please consult with a healthcare professional.
            """)
        else:
            st.success(f"Based on the AQ test, there is a {probability*100:.2f}% probability of autism spectrum traits.")
            st.info("""
            **Note:** This is a screening tool and not a diagnostic test. 
            If you have concerns, please consult with a healthcare professional.
            """)
        
        # Display total score
        st.metric("Total AQ Score", total_score)

if __name__ == "__main__":
    main() 