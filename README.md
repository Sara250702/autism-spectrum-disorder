# Autism Detection System

This is a Streamlit web application that uses the Autism Spectrum Quotient (AQ) test to help assess the likelihood of autism spectrum traits. The application uses a machine learning model trained on the AQ test data to provide predictions.

## Features

- User-friendly interface for taking the AQ test
- 10 carefully selected questions from the AQ test
- Collects relevant demographic information
- Provides probability-based predictions
- Displays total AQ score
- Responsive design with a clean, modern interface

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure you have the trained model file (`model.pkl`) in the same directory as `app.py`
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:8501)

## How to Use

1. Fill in your demographic information in the sidebar
2. Answer all 10 questions honestly
3. Click the "Get Prediction" button to see your results
4. Review your total AQ score and prediction probability

## Important Note

This application is a screening tool and not a diagnostic test. The results should not be used as a substitute for professional medical advice, diagnosis, or treatment. If you have concerns about autism spectrum traits, please consult with a healthcare professional.

## Requirements

- Python 3.8 or higher
- Dependencies listed in requirements.txt
- Trained model file (model.pkl) 