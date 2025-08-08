import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import gdown
import os



# Set page config
st.set_page_config(
    page_title="Airline Satisfaction Predictor",
    page_icon="",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1E90FF 0%, #00BFFF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .prediction-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .feature-importance {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and preprocessing objects 
@st.cache_data
def load_artifacts():
    # Google Drive file IDs
    model_file_id = '1998nsWAw5qyroBBOPVn20geDze-hknxD'
    scaler_file_id = '1EFpNaC57_97TBj3gCCv7EGPAH6IVyfFd'  
    encoder_file_id = '1mumuM82MhlnPhwggAPCnAqXIONzAsTIy'  
    
    # Download files from Google Drive
    def download_file(file_id, output):
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, output, quiet=False)
            return True
        except Exception as e:
            st.error(f"Failed to download {output}: {str(e)}")
            return False
    
    # Download files with progress indicators
    with st.spinner('Loading model files...'):
        if not os.path.exists('best_airline_model.pkl'):
            if not download_file(model_file_id, 'best_airline_model.pkl'):
                st.stop()
        if not os.path.exists('airline_scaler.pkl'):
            if not download_file(scaler_file_id, 'airline_scaler.pkl'):
                st.stop()
        if not os.path.exists('airline_label_encoder.pkl'):
            if not download_file(encoder_file_id, 'airline_label_encoder.pkl'):
                st.stop()
    
    # Load the files
    try:
        model = joblib.load('best_airline_model.pkl')
        scaler = joblib.load('airline_scaler.pkl')
        encoder = joblib.load('airline_label_encoder.pkl')
    except Exception as e:
        st.error(f"Failed to load model files: {str(e)}")
        st.stop()
    
    # Rest of your function remains the same...
    # [Keep all your feature extraction code]
    
    return model, scaler, encoder, original_features
I

model, scaler, encoder, EXPECTED_FEATURES = load_artifacts()

# Header
st.markdown('<h1 class="main-header"> Airline Passenger Satisfaction Predictor</h1>', unsafe_allow_html=True)
st.markdown("### Predict passenger satisfaction based on flight experience")

# Create tabs
tab1, tab2 = st.tabs([" Make Prediction", " Model Insights"])

with tab1:
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Passenger Details")
            age = st.slider("Age", 7, 85, 35)
            gender = st.selectbox("Gender", ["Female", "Male"])
            customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
            
        with col2:
            st.subheader("Flight Details")
            flight_distance = st.slider("Flight Distance (miles)", 0, 5000, 1000)
            departure_delay = st.slider("Departure Delay (minutes)", 0, 300, 0)
            arrival_delay = st.slider("Arrival Delay (minutes)", 0, 300, 0)
            travel_type = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
            flight_class = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
            
        with col3:
            st.subheader("Service Ratings (1-5)")
            seat_comfort = st.select_slider("Seat comfort", options=[1,2,3,4,5], value=3)
            departure_arrival_convenience = st.select_slider("Departure/Arrival time convenient", options=[1,2,3,4,5], value=3)
            food_drink = st.select_slider("Food and drink", options=[1,2,3,4,5], value=3)
            gate_location = st.select_slider("Gate location", options=[1,2,3,4,5], value=3)
            inflight_wifi = st.select_slider("Inflight wifi service", options=[1,2,3,4,5], value=3)
            inflight_entertainment = st.select_slider("Inflight entertainment", options=[1,2,3,4,5], value=3)
            inflight_service = st.select_slider("Inflight service", options=[1,2,3,4,5], value=3)
        
        submitted = st.form_submit_button("Predict Satisfaction")

    if submitted:
        # Prepare input data
        input_data = {
            'Unnamed: 0': 0,  # Dummy value for the index column
            'Age': age,
            'Flight Distance': flight_distance,
            'Departure Delay in Minutes': departure_delay,
            'Arrival Delay in Minutes': arrival_delay,
            'Gender_Female': 1 if gender == "Female" else 0,
            'Gender_Male': 1 if gender == "Male" else 0,
            'Customer Type_Loyal Customer': 1 if customer_type == "Loyal Customer" else 0,
            'Customer Type_disloyal Customer': 1 if customer_type == "disloyal Customer" else 0,
            'Type of Travel_Business travel': 1 if travel_type == "Business travel" else 0,
            'Type of Travel_Personal Travel': 1 if travel_type == "Personal Travel" else 0,
            'Class_Business': 1 if flight_class == "Business" else 0,
            'Class_Eco': 1 if flight_class == "Eco" else 0,
            'Class_Eco Plus': 1 if flight_class == "Eco Plus" else 0,
            'Seat comfort': seat_comfort,
            'Departure/Arrival time convenient': departure_arrival_convenience,
            'Food and drink': food_drink,
            'Gate location': gate_location,
            'Inflight wifi service': inflight_wifi,
            'Inflight entertainment': inflight_entertainment,
            'Inflight service': inflight_service,
            'Ease of Online booking': 3,
            'On-board service': 3,
            'Leg room service': 3,
            'Baggage handling': 3,
            'Checkin service': 3,
            'Cleanliness': 3,
            'Online boarding': 3
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])[EXPECTED_FEATURES]
        
        try:
            # Scale and predict
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]
            prediction_proba = model.predict_proba(scaled_input)[0]
            
            # Display results
            with st.container():
                st.markdown("### Prediction Results")
                if prediction == 1:
                    st.success(f" Predicted Satisfaction: Satisfied ({prediction_proba[1]*100:.1f}% confidence)")
                else:
                    st.error(f" Predicted Satisfaction: Neutral/Dissatisfied ({prediction_proba[0]*100:.1f}% confidence)")
                
                # Show probability breakdown
                st.write("Probability Breakdown:")
                proba_df = pd.DataFrame({
                    "Satisfaction Level": encoder.classes_,
                    "Probability": prediction_proba
                })
                st.bar_chart(proba_df.set_index("Satisfaction Level"))
        
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            with st.expander("Debug Info"):
                st.write("Features expected by model:", EXPECTED_FEATURES)
                st.write("Features provided:", input_df.columns.tolist())
                st.write("First row of data:", input_df.iloc[0].to_dict())

with tab2:
    st.subheader("Model Performance and Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("###  Best Model Performance")
        st.markdown("""
        - **Model Type**: Random Forest
        - **Accuracy**: 94.3%
        - **Precision**: 94.1%
        - **Recall**: 94.3%
        - **F1 Score**: 94.2%
        """)
        
        st.markdown("###  Key Findings")
        st.markdown("""
        - Flight delays (both departure and arrival) are top predictors of dissatisfaction
        - Business class passengers report 40% higher satisfaction
        - Seat comfort and inflight service ratings are critical
        - Cleanliness scores strongly correlate with overall satisfaction
        """)
    
    with col2:
        st.markdown("###  Feature Importance")
        if hasattr(model, 'feature_importances_'):
            try:
                # Ensure we only use available features
                valid_features = [f for f in EXPECTED_FEATURES if f != 'Unnamed: 0']
                importance_scores = model.feature_importances_[:len(valid_features)]
                
                importance_df = pd.DataFrame({
                    'Feature': valid_features,
                    'Importance': importance_scores
                }).sort_values('Importance', ascending=False).head(10)
                
                fig = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title='Top 10 Most Important Features',
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                
                with st.expander("View as list"):
                    for idx, row in importance_df.iterrows():
                        st.markdown(f"- **{row['Feature']}**: {row['Importance']:.3f}")
                        
            except Exception as e:
                st.warning(f"Couldn't visualize feature importance: {str(e)}")
                st.markdown("""
                **Top influential factors** (based on model analysis):
                1. Flight Distance
                2. Departure/Arrival Delays  
                3. Seat Comfort Ratings
                4. Inflight Service Quality
                5. Cleanliness Scores
                """)
        else:
            st.info("Feature importance not available for this model type")
            st.markdown("""
            **Key influential factors** (from EDA):
            1. Flight Distance
            2. Departure/Arrival Delays  
            3. Seat Comfort Ratings
            4. Inflight Service Quality
            5. Cleanliness Scores
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
     Airline Passenger Satisfaction Predictor | Built with Streamlit | 
    Data Source: Kaggle Airline Satisfaction Dataset
</div>
""", unsafe_allow_html=True)
