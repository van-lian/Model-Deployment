import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import requests
from io import BytesIO

# Set page config FIRST - before any other Streamlit commands
st.set_page_config(
    page_title="Hotel Booking Predictor",
    page_icon="🏨",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        margin-top: 20px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    # SAS URLs for all files
    model_url = "https://forproject112.blob.core.windows.net/model-deployment/random_forest_model.pkl?sp=r&st=2025-04-18T13:08:16Z&se=2025-04-18T21:08:16Z&spr=https&sv=2024-11-04&sr=b&sig=XHlra8KjScUjjphJyE7NOmTg2wbZvwiKfy8e%2FHGWj1I%3D"
    scaler_url = "https://forproject112.blob.core.windows.net/model-deployment/scaler.pkl?sp=r&st=2025-04-18T13:08:40Z&se=2025-04-18T21:08:40Z&spr=https&sv=2024-11-04&sr=b&sig=5VNhwnLBDDIN%2FytBik0sAOu3kHO%2FcSTX3il5nN4XKw0%3D"
    features_url = "https://forproject112.blob.core.windows.net/model-deployment/feature_names.pkl?sp=r&st=2025-04-18T13:07:54Z&se=2025-04-18T21:07:54Z&spr=https&sv=2024-11-04&sr=b&sig=r6q6sPHo8tkfBVDfQtp%2FMgNULb93hzQsp%2FtETv%2FIFEA%3D"
    
    try:
        # Download and load the model
        model_response = requests.get(model_url)
        model_response.raise_for_status()
        model = pickle.load(BytesIO(model_response.content))
        
        # Download and load the scaler
        scaler_response = requests.get(scaler_url)
        scaler_response.raise_for_status()
        scaler = pickle.load(BytesIO(scaler_response.content))
        
        # Download and load the features
        features_response = requests.get(features_url)
        features_response.raise_for_status()
        features = pickle.load(BytesIO(features_response.content))
        
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading components from Azure: {str(e)}")
        return None, None, None

# Load the model components
model, scaler, features = load_model()

# Store model components in session state
if model is not None:
    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.features = features
else:
    st.error("Failed to load model. Please check your Azure connection and SAS URL.")
    st.stop()

def predict_booking_status(input_data):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for feature in st.session_state.features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Select and order features
        input_df = input_df[st.session_state.features]
        
        # Scale the features
        scaled_features = st.session_state.scaler.transform(input_df)
        
        # Make prediction
        prediction = st.session_state.model.predict(scaled_features)
        prediction_proba = st.session_state.model.predict_proba(scaled_features)
        
        return prediction[0], prediction_proba[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Main UI
st.title('🏨 Hotel Booking Cancellation Predictor')
st.write('This app predicts whether a hotel booking is likely to be cancelled.')

# Create tabs for different test cases
tab1, tab2 = st.tabs(["Business Traveler", "Family Vacation"])

with tab1:
    st.header("💼 Business Traveler Booking")
    st.write("Single business traveler making a short-term booking.")
    
    col1, col2 = st.columns(2)
    with col1:
        adults = st.number_input("Number of Adults", value=1, min_value=1, max_value=10, key='b_adults')
        lead_time = st.number_input("Lead Time (days)", value=7, min_value=0, max_value=365, key='b_lead')
        room_type = st.selectbox("Room Type", ['Standard', 'Deluxe', 'Executive', 'Presidential'], key='b_room')
    with col2:
        nights = st.number_input("Number of Nights", value=2, min_value=1, max_value=30, key='b_nights')
        price = st.number_input("Price per Night (€)", value=120, min_value=50, max_value=1000, key='b_price')
        meal_plan = st.selectbox("Meal Plan", ['Room Only', 'Breakfast', 'Half Board', 'Full Board'], key='b_meal')
    
    if st.button('Predict Booking Status', key='b_predict'):
        # Map inputs to model features
        test_case = {
            'no_of_adults': adults,
            'no_of_children': 0,
            'no_of_weekend_nights': min(nights, 2),
            'no_of_week_nights': max(0, nights-2),
            'required_car_parking_space': 0,
            'lead_time': lead_time,
            'arrival_year': 2024,
            'arrival_month': 3,
            'arrival_date': 15,
            'repeated_guest': 0,
            'no_of_previous_cancellations': 0,
            'no_of_previous_bookings_not_canceled': 1,
            'avg_price_per_room': price,
            'no_of_special_requests': 1,
            'total_nights': nights,
            'total_guests': adults,
            'total_previous_bookings': 1,
            'type_of_meal_plan_encoded': ['Room Only', 'Breakfast', 'Half Board', 'Full Board'].index(meal_plan),
            'room_type_reserved_encoded': ['Standard', 'Deluxe', 'Executive', 'Presidential'].index(room_type),
            'market_segment_type_encoded': 2  # Business segment
        }
        
        prediction, probabilities = predict_booking_status(test_case)
        
        if prediction is not None:
            st.subheader('Prediction Results:')
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error('❌ High Risk of Cancellation')
                else:
                    st.success('✅ Low Risk of Cancellation')
            
            with col2:
                st.metric(
                    "Probability of Cancellation",
                    f"{probabilities[1]:.1%}",
                    delta=f"{0.5 - probabilities[1]:.1%}",
                    delta_color="inverse"
                )

with tab2:
    st.header("👨‍👩‍👧‍👦 Family Vacation Booking")
    st.write("Family group making a vacation booking during peak season.")
    
    col1, col2 = st.columns(2)
    with col1:
        adults = st.number_input("Number of Adults", value=2, min_value=1, max_value=10, key='f_adults')
        children = st.number_input("Number of Children", value=2, min_value=0, max_value=10, key='f_children')
        lead_time = st.number_input("Lead Time (days)", value=90, min_value=0, max_value=365, key='f_lead')
    with col2:
        nights = st.number_input("Number of Nights", value=7, min_value=1, max_value=30, key='f_nights')
        price = st.number_input("Price per Night (€)", value=200, min_value=50, max_value=1000, key='f_price')
        meal_plan = st.selectbox("Meal Plan", ['Room Only', 'Breakfast', 'Half Board', 'Full Board'], key='f_meal')
    
    parking = st.checkbox("Requires Parking", value=True, key='f_parking')
    room_type = st.selectbox("Room Type", ['Standard', 'Deluxe', 'Executive', 'Presidential'], key='f_room')
    
    if st.button('Predict Booking Status', key='f_predict'):
        # Map inputs to model features
        test_case = {
            'no_of_adults': adults,
            'no_of_children': children,
            'no_of_weekend_nights': min(nights, 2),
            'no_of_week_nights': max(0, nights-2),
            'required_car_parking_space': 1 if parking else 0,
            'lead_time': lead_time,
            'arrival_year': 2024,
            'arrival_month': 7,
            'arrival_date': 1,
            'repeated_guest': 0,
            'no_of_previous_cancellations': 0,
            'no_of_previous_bookings_not_canceled': 0,
            'avg_price_per_room': price,
            'no_of_special_requests': 2,
            'total_nights': nights,
            'total_guests': adults + children,
            'total_previous_bookings': 0,
            'type_of_meal_plan_encoded': ['Room Only', 'Breakfast', 'Half Board', 'Full Board'].index(meal_plan),
            'room_type_reserved_encoded': ['Standard', 'Deluxe', 'Executive', 'Presidential'].index(room_type),
            'market_segment_type_encoded': 1  # Leisure segment
        }
        
        prediction, probabilities = predict_booking_status(test_case)
        
        if prediction is not None:
            st.subheader('Prediction Results:')
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error('❌ High Risk of Cancellation')
                else:
                    st.success('✅ Low Risk of Cancellation')
            
            with col2:
                st.metric(
                    "Probability of Cancellation",
                    f"{probabilities[1]:.1%}",
                    delta=f"{0.5 - probabilities[1]:.1%}",
                    delta_color="inverse"
                )

# Add sidebar with feature explanations and insights
with st.sidebar:
    st.header("ℹ️ About")
    st.write("""
    This prediction model uses Random Forest algorithm with 90.38% accuracy to predict hotel booking cancellations.
    """)
    
    st.header("📊 Key Factors")
    st.write("""
    The model considers these important factors:
    - Lead time before arrival
    - Room price
    - Length of stay
    - Number of guests
    - Previous booking history
    - Special requests
    """)
    
    st.header("💡 Tips to Reduce Cancellation Risk")
    st.write("""
    1. Book closer to arrival date
    2. Make special requests
    3. Choose flexible room types
    4. Consider meal plan options
    5. Book during off-peak seasons
    """)

    st.header("🎯 Model Performance")
    st.write("""
    - Accuracy: 90.38%
    - Cross-validation score: 89.52%
    - Balanced performance for both cancellation and non-cancellation predictions
    """)