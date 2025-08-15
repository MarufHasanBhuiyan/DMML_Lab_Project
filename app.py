import streamlit as st
import pandas as pd
import joblib
import numpy as np

# # Disable WebSocket compression and force static frontend
# st.set_option('server.enableWebsocketCompression', False)
# st.set_option('server.enableCORS', False)
# st.set_option('server.enableXsrfProtection', False)

# Set page configuration
st.set_page_config(page_title="Multi-Outcome Prediction App", layout="wide", page_icon="📊")

# Custom CSS for beautiful UI with visible green hints
st.markdown("""
<style>
    .main {
        background: linear-gradient(to bottom, #d9e6ff, #f5faff);
        padding: 20px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 12px 30px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.15);
    }
    .stButton>button:hover {
        background-color: #3d8b40;
        transform: scale(1.1);
        box-shadow: 0 6px 12px rgba(0,0,0,0.25);
    }
    .stSelectbox, .stNumberInput {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSelectbox label, .stNumberInput label {
        color: #1a2526 !important;
        font-weight: 600;
        font-size: 16px;
    }
    .header {
        color: #1a2526;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subheader {
        color: #1a2526;
        font-size: 24px;
        font-weight: 600;
        margin-top: 15px;
    }
    .hint {
        color: #4CAF50 !important;
        font-size: 14px !important;
        font-weight: bold !important;
        margin-bottom: 5px;
    }
    .prediction-box {
        background-color: #e6f3ff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #1a2526;
        font-size: 18px;
    }
    .stMarkdown, .stText {
        color: #1a2526 !important;
    }
    .stAlert, .stAlert * {
        color: #1a2526 !important;
        background-color: #ffe6e6 !important;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and author
st.markdown("<div class='header'>Multi-Outcome Prediction App</div>", unsafe_allow_html=True)
st.markdown("**Author**: Maruf Hasan Bhuiyan", unsafe_allow_html=True)
st.markdown("Enter values to predict **Success Label**, **Sleep Disorder**, and **Final Grade (G3)**", unsafe_allow_html=True)

# Load models and scaler
try:
    model_success = joblib.load('model_success.pkl')
    model_sleep = joblib.load('model_sleep.pkl')
    model_g3 = joblib.load('model_g3.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# Define the 22 selected features for models
selected_features = ['Sleep Duration', 'Occupation_Nurse', 'Daily Steps', 'Quality of Sleep', 
                    'Physical Activity Level', 'net_worth', 'Systolic', 'annual_income', 
                    'Heart Rate', 'BMI Category_Overweight', 'job_satisfaction', 'Diastolic', 
                    'absences', 'goout', 'work_life_balance', 'BMI Category_Normal', 
                    'home_ownership_renting', 'career_growth', 'failures', 'years_in_field', 
                    'social_connections', 'age']

# Define the 21 numeric features for scaler (in exact order from Colab)
numeric_cols = ['Gender', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 
                'Stress Level', 'Heart Rate', 'Daily Steps', 'Systolic', 'Diastolic', 
                'age', 'studytime', 'failures', 'goout', 'absences', 'annual_income', 
                'net_worth', 'job_satisfaction', 'work_life_balance', 'career_growth', 
                'social_connections', 'years_in_field']

# Define categorical features for dropdowns
categorical_cols = ['Occupation_Nurse', 'BMI Category_Overweight', 'BMI Category_Normal', 
                    'home_ownership_renting']

# Define options for categorical features
categorical_options = {
    'Occupation_Nurse': ['No', 'Yes'],
    'BMI Category_Overweight': ['No', 'Yes'],
    'BMI Category_Normal': ['No', 'Yes'],
    'home_ownership_renting': ['No', 'Yes']
}

# Define ranges for numeric features
numeric_ranges = {
    'Gender': {'min': 0, 'max': 1, 'default': 0, 'hint': 'Select 0 for Male, 1 for Female'},
    'Sleep Duration': {'min': 4.0, 'max': 10.0, 'default': 7.0, 'hint': 'Enter hours of sleep (Range: 4–10)'},
    'Quality of Sleep': {'min': 1, 'max': 10, 'default': 6, 'hint': 'Enter sleep quality (Range: 1–10)'},
    'Physical Activity Level': {'min': 0, 'max': 100, 'default': 50, 'hint': 'Enter activity level (Range: 0–100)'},
    'Stress Level': {'min': 1, 'max': 10, 'default': 5, 'hint': 'Enter stress level (Range: 1–10)'},
    'Heart Rate': {'min': 50, 'max': 100, 'default': 70, 'hint': 'Enter heart rate (Range: 50–100 bpm)'},
    'Daily Steps': {'min': 3000, 'max': 15000, 'default': 8000, 'hint': 'Enter steps per day (Range: 3000–15000)'},
    'Systolic': {'min': 90, 'max': 180, 'default': 120, 'hint': 'Enter systolic BP (Range: 90–180)'},
    'Diastolic': {'min': 60, 'max': 120, 'default': 80, 'hint': 'Enter diastolic BP (Range: 60–120)'},
    'age': {'min': 15, 'max': 65, 'default': 25, 'hint': 'Enter age (Range: 15–65)'},
    'studytime': {'min': 1, 'max': 4, 'default': 2, 'hint': 'Enter study time (Range: 1–4, 1=<2h, 2=2-5h, 3=5-10h, 4=>10h)'},
    'failures': {'min': 0, 'max': 4, 'default': 0, 'hint': 'Enter past class failures (Range: 0–4)'},
    'goout': {'min': 1, 'max': 5, 'default': 3, 'hint': 'Enter going out frequency (Range: 1–5)'},
    'absences': {'min': 0, 'max': 75, 'default': 5, 'hint': 'Enter school absences (Range: 0–75)'},
    'annual_income': {'min': 10000, 'max': 100000, 'default': 40000, 'hint': 'Enter annual income (Range: $10K–$100K)'},
    'net_worth': {'min': -50000, 'max': 500000, 'default': 50000, 'hint': 'Enter net worth (Range: -$50K–$500K)'},
    'job_satisfaction': {'min': 1, 'max': 10, 'default': 5, 'hint': 'Enter job satisfaction (Range: 1–10)'},
    'work_life_balance': {'min': 1, 'max': 10, 'default': 5, 'hint': 'Enter work-life balance (Range: 1–10)'},
    'career_growth': {'min': 1, 'max': 10, 'default': 5, 'hint': 'Enter career growth score (Range: 1–10)'},
    'social_connections': {'min': 0, 'max': 50, 'default': 10, 'hint': 'Enter number of connections (Range: 0–50)'},
    'years_in_field': {'min': 0, 'max': 30, 'default': 5, 'hint': 'Enter years in field (Range: 0–30)'}
}

# Create input form
st.markdown("<div class='header'>Input Features</div>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 1])

# Categorical inputs
with col1:
    st.markdown("<div class='subheader'>Categorical Features</div>", unsafe_allow_html=True)
    input_data = {}
    for feature in categorical_cols:
        st.markdown(f"<div class='hint'>Hint: Select {feature.replace('_', ' ')} (Yes/No)</div>", unsafe_allow_html=True)
        value = st.selectbox(f"{feature.replace('_', ' ')}", 
                             options=categorical_options[feature])
        input_data[feature] = 1 if value == 'Yes' else 0

# Numeric inputs
with col2:
    st.markdown("<div class='subheader'>Numeric Features</div>", unsafe_allow_html=True)
    for feature in numeric_cols:
        st.markdown(f"<div class='hint'>Hint: {numeric_ranges[feature]['hint']}</div>", unsafe_allow_html=True)
        value = st.number_input(f"{feature.replace('_', ' ')}", 
                               min_value=float(numeric_ranges[feature]['min']),
                               max_value=float(numeric_ranges[feature]['max']),
                               value=float(numeric_ranges[feature]['default']),
                               placeholder=numeric_ranges[feature]['hint'])
        input_data[feature] = value

# Create input DataFrame for scaler (all 21 numeric features)
input_df_numeric = pd.DataFrame([input_data], columns=numeric_cols)

# Standardize numeric inputs
try:
    input_numeric = scaler.transform(input_df_numeric)
    input_df_numeric[numeric_cols] = input_numeric
except Exception as e:
    st.error(f"Scaler Error: {e}")
    st.stop()

# Create input DataFrame for models (22 selected features)
input_df = pd.DataFrame([input_data], columns=selected_features)

# Copy standardized values to input_df for numeric features in selected_features
for col in numeric_cols:
    if col in selected_features:
        input_df[col] = input_df_numeric[col]

# Predict button
if st.button("Predict Outcomes", type="primary", key="predict_button"):
    try:
        # Make predictions
        pred_success = model_success.predict(input_df)[0]
        pred_sleep = model_sleep.predict(input_df)[0]
        pred_g3 = model_g3.predict(input_df)[0]

        # Display predictions
        st.markdown("<div class='header'>Predictions</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-box'><b>Success Label</b>: {'Successful' if pred_success == 1 else 'Not Successful'}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-box'><b>Sleep Disorder</b>: {pred_sleep}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-box'><b>Final Grade (G3)</b>: {pred_g3:.1f}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction Error: {e}")

# Optional: Show input data
if st.checkbox("Show Input Data"):
    st.markdown("<div class='subheader'>Input Data</div>", unsafe_allow_html=True)
    st.dataframe(input_df.style.set_properties(**{'background-color': '#e6f3ff', 'border-radius': '5px', 'color': '#1a2526'}))