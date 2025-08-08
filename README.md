# Airline Passenger Satisfaction Prediction

## Project Overview
This project predicts airline passenger satisfaction using machine learning. It analyzes flight experience features like service ratings, delays, and travel class from a Kaggle dataset of 100,000+ passenger surveys. The Random Forest model performed best with 96% accuracy. The solution includes an interactive dashboard for exploring data and making predictions.

## Features 
- **Interactive Dashboard**:
- **Data Exploration**: 
  - Dataset cleaning and preprocessing
  - Missing value handling
  - Feature correlation analysis
- **Passenger Analysis**:
  - Satisfaction distribution
  - Service rating visualizations
  - Delay impact analysis
- **Model Training**:
  - Multiple ML models (Random Forest, SVM, KNN, etc.)
  - Performance metrics comparison
- **Satisfaction Prediction**:
  - Interactive input form
  - Confidence score display
  - Key factors explanation

## Technologies Used:
- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**:
  - Scikit-learn
  - Random Forest
- **Data Visualization**:
  - Matplotlib
  - Seaborn
  - Plotly
- **Data Processing**: Pandas, NumPy

## Installation 
Clone the repository:
```bash
git clone https://github.com/AyushmanShrestha2000/Airline-Customer-Satisfaction
cd airline-satisfaction-prediction

## Install dependencies:
- pip install -r requirements.txt

## Run the application:
- streamlit run app.py

## File Structure:
airline-satisfaction-prediction/
├── app.py # Main Streamlit application
├── Airline_train.csv # Passenger dataset
├── README.md # This file
├── requirements.txt # Python dependencies

Data source: [Kaggle Airline Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
