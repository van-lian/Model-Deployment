import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# Setup error handling and file path verification
def load_model_files():
    try:
        # Get the absolute path of the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define file paths
        model_path = os.path.join(current_dir, 'random_forest_model.pkl')
        scaler_path = os.path.join(current_dir, 'scaler.pkl')
        features_path = os.path.join(current_dir, 'feature_names.pkl')
        
        # Verify files exist
        if not all(os.path.exists(f) for f in [model_path, scaler_path, features_path]):
            st.error("Required model files not found. Please check if all files are in the correct location.")
            return None, None, None
        
        # Load the files
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        with open(features_path, 'rb') as file:
            features = pickle.load(file)
            
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None, None

def predict_booking_status(input_data, model, scaler, features):
    try:
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for feature in features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Select and order features
        input_df = input_df[features]
        
        # Scale the features
        scaled_features = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)
        
        return prediction[0], prediction_proba[0]
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None

# Streamlit UI
st.title('Hotel Booking Cancellation Predictor')
st.write('This app predicts whether a hotel booking is likely to be cancelled.')

# Load model files
model, scaler, features = load_model_files()

if model is None or scaler is None or features is None:
    st.error("Cannot proceed without required model files.")
    st.stop()

# Create tabs for different test cases
tab1, tab2 = st.tabs(["Test Case 1", "Test Case 2"])

with tab1:
    st.header("Business Traveler Booking")
    st.write("This test case represents a typical business traveler booking.")
    
    if st.button('Run Test Case 1'):
        test_case_1 = {
            'no_of_adults': 1,
            'no_of_children': 0,
            'no_of_weekend_nights': 0,
            'no_of_week_nights': 2,
            'required_car_parking_space': 0,
            'lead_time': 7,
            'arrival_year': 2024,
            'arrival_month': 3,
            'arrival_date': 15,
            'repeated_guest': 0,
            'no_of_previous_cancellations': 0,
            'no_of_previous_bookings_not_canceled': 1,
            'avg_price_per_room': 120,
            'no_of_special_requests': 1,
            'total_nights': 2,
            'total_guests': 1,
            'total_previous_bookings': 1,
            'type_of_meal_plan_encoded': 1,
            'room_type_reserved_encoded': 2,
            'market_segment_type_encoded': 2
        }
        
        prediction, probabilities = predict_booking_status(test_case_1, model, scaler, features)
        
        if prediction is not None and probabilities is not None:
            st.subheader('Prediction Results:')
            if prediction == 1:
                st.error('❌ Booking likely to be cancelled')
            else:
                st.success('✅ Booking likely to be confirmed')
                
            st.write(f'Probability of cancellation: {probabilities[1]:.2%}')
            st.write(f'Probability of confirmation: {probabilities[0]:.2%}')

with tab2:
    st.header("Family Vacation Booking")
    st.write("This test case represents a family vacation booking during peak season.")
    
    if st.button('Run Test Case 2'):
        test_case_2 = {
            'no_of_adults': 2,
            'no_of_children': 2,
            'no_of_weekend_nights': 2,
            'no_of_week_nights': 5,
            'required_car_parking_space': 1,
            'lead_time': 90,
            'arrival_year': 2024,
            'arrival_month': 7,
            'arrival_date': 1,
            'repeated_guest': 0,
            'no_of_previous_cancellations': 0,
            'no_of_previous_bookings_not_canceled': 0,
            'avg_price_per_room': 200,
            'no_of_special_requests': 2,
            'total_nights': 7,
            'total_guests': 4,
            'total_previous_bookings': 0,
            'type_of_meal_plan_encoded': 2,
            'room_type_reserved_encoded': 3,
            'market_segment_type_encoded': 1
        }
        
        prediction, probabilities = predict_booking_status(test_case_2, model, scaler, features)
        
        if prediction is not None and probabilities is not None:
            st.subheader('Prediction Results:')
            if prediction == 1:
                st.error('❌ Booking likely to be cancelled')
            else:
                st.success('✅ Booking likely to be confirmed')
                
            st.write(f'Probability of cancellation: {probabilities[1]:.2%}')
            st.write(f'Probability of confirmation: {probabilities[0]:.2%}')

# Add sidebar with feature explanations
with st.sidebar:
    st.header("Feature Information")
    st.write("""
    The model considers various features including:
    - Number of adults and children
    - Length of stay (weekend and weekday nights)
    - Lead time before arrival
    - Room price
    - Special requests
    - Previous booking history
    - Meal plan type
    - Room type
    - Market segment
    """)
    
    st.header("Model Performance")
    st.write("""
    The Random Forest model achieves:
    - Accuracy: 90.38%
    - Robust cross-validation performance
    - Strong predictive power for both cancellations and confirmations
    """)