import streamlit as st
import numpy as np
from aq_test import load_model_and_encoders as load_aq_model
from video_screening import load_model as load_video_model

def combine_predictions(aq_probability, video_behavior, video_confidence):
    """
    Combine predictions from AQ test and video screening.
    Returns a final probability score and interpretation.
    """
    # Convert video confidence to a probability-like score (0-1)
    video_score = video_confidence / 100.0
    
    # Weight the predictions (can be adjusted based on validation)
    aq_weight = 0.6  # AQ test is given more weight
    video_weight = 0.4
    
    # Calculate combined probability
    combined_prob = (aq_probability * aq_weight) + (video_score * video_weight)
    
    # Determine interpretation
    if combined_prob >= 0.7:
        interpretation = "High likelihood of autism spectrum traits"
    elif combined_prob >= 0.4:
        interpretation = "Moderate likelihood of autism spectrum traits"
    else:
        interpretation = "Low likelihood of autism spectrum traits"
    
    return combined_prob, interpretation

def get_multimodal_prediction(aq_results, video_results):
    """
    Process results from both AQ test and video screening.
    
    Args:
        aq_results: Dictionary containing AQ test results
        video_results: Dictionary containing video screening results
    """
    # Extract AQ test probability
    aq_probability = aq_results.get('probability', 0)
    
    # Extract video results
    video_behavior = video_results.get('behavior', 'none')
    video_confidence = video_results.get('confidence', 0)
    
    # Combine predictions
    combined_prob, interpretation = combine_predictions(
        aq_probability, video_behavior, video_confidence
    )
    
    return {
        'combined_probability': combined_prob,
        'interpretation': interpretation,
        'aq_probability': aq_probability,
        'video_behavior': video_behavior,
        'video_confidence': video_confidence
    }

def display_results(results):
    """
    Display the multimodal prediction results in a formatted way.
    """
    st.markdown("""
    <div style='background-color: #ffffff; padding: 20px; border-radius: 10px; margin-top: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
        <h2 style='color: #2c3e50; text-align: center;'>Multimodal Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Display individual results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### AQ Test Results")
        st.metric(
            "Probability Score",
            f"{results['aq_probability']*100:.2f}%"
        )
    
    with col2:
        st.markdown("### Video Analysis Results")
        st.metric(
            "Detected Behavior",
            results['video_behavior']
        )
        st.metric(
            "Confidence",
            f"{results['video_confidence']:.2f}%"
        )
    
    # Display combined results
    st.markdown("### Combined Analysis")
    st.metric(
        "Overall Probability",
        f"{results['combined_probability']*100:.2f}%"
    )
    
    # Display interpretation
    if results['combined_probability'] >= 0.7:
        st.error(results['interpretation'])
    elif results['combined_probability'] >= 0.4:
        st.warning(results['interpretation'])
    else:
        st.success(results['interpretation'])
    
    # Add explanation
    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 15px; border-radius: 10px; margin: 10px 0;'>
        <p style='color: #2c3e50;'>
            <strong>Note:</strong> This combined analysis provides a more comprehensive assessment by integrating both behavioral observations and questionnaire responses. However, this is not a diagnostic tool and should be used in conjunction with professional evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True) 