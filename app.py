import streamlit as st
from aq_test import main as aq_test_main
from video_screening import main as video_screening_main
from multimodal import get_multimodal_prediction, display_results

# Set page config
st.set_page_config(
    page_title="Autism Detection System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize all session state variables
if 'aq_results' not in st.session_state:
    st.session_state.aq_results = None
if 'video_results' not in st.session_state:
    st.session_state.video_results = None
if 'answers' not in st.session_state:
    st.session_state.answers = [0] * 10  # Initialize with 10 zeros for AQ test answers

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
    </style>
    """, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("<h1 style='text-align: center;'>🧠 Autism Detection System</h1>", unsafe_allow_html=True)
    
    # Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["AQ Test", "Video Screening", "Multimodal Prediction"]
    )
    
    # Page content
    if page == "AQ Test":
        st.markdown("<h2 style='text-align: center;'>📝 Autism Spectrum Quotient (AQ) Test</h2>", unsafe_allow_html=True)
        aq_test_main()
        
    elif page == "Video Screening":
        st.markdown("<h2 style='text-align: center;'>🎥 Video Behavior Analysis</h2>", unsafe_allow_html=True)
        video_screening_main()
        
    elif page == "Multimodal Prediction":
        st.markdown("<h2 style='text-align: center;'>🔍 Multimodal Analysis</h2>", unsafe_allow_html=True)
        
        # Check if both tests have been completed
        if st.session_state.aq_results is None or st.session_state.video_results is None:
            st.warning("Please complete both the AQ Test and Video Screening before viewing the multimodal analysis.")
            return
        
        # Get and display multimodal prediction
        results = get_multimodal_prediction(
            st.session_state.aq_results,
            st.session_state.video_results
        )
        display_results(results)

if __name__ == "__main__":
    main() 