# Airline Passenger Satisfaction Analysis & Prediction System

![Airline Satisfaction Dashboard](https://img.freepik.com/free-vector/airport-skyline-with-airplanes_107791-1783.jpg)

## 📌 Project Overview

This project provides a complete pipeline for:
1. **Exploratory Data Analysis** of airline passenger satisfaction data
2. **Machine Learning Modeling** to predict passenger satisfaction
3. **Web Application** for real-time predictions via Streamlit

## 📂 Project Structure
airline-satisfaction/
├── notebooks/
│ └── Airline_Passenger_Analysis.ipynb # Jupyter notebook with full analysis
├── app/
│ └── streamlit_app.py # Streamlit web application
├── models/
│ ├── best_airline_model.pkl # Trained model
│ ├── airline_scaler.pkl # Feature scaler
│ └── airline_label_encoder.pkl # Label encoder
├── data/
│ └── Airline_train.csv # Original dataset
└── README.md


## 🔍 Data Analysis Notebook

### Key Features
- Comprehensive EDA with visualizations
- Data cleaning pipeline
- Feature engineering
- Model comparison (10+ algorithms)
- Hyperparameter tuning
- Model evaluation metrics


### Notebook Contents
1. **Data Exploration**
   - Dataset statistics
   - Missing value analysis
   - Target variable distribution

2. **Visualizations**
   - Satisfaction distribution pie chart
   - Flight distance vs satisfaction
   - Delay impact analysis
   - Correlation heatmap

3. **Machine Learning**
   - Logistic Regression
   - Random Forest
   - SVM variants
   - KNN with optimal k
   - Naive Bayes
   - Decision Trees
   - K-Means clustering

4. **Results**
   - Cross-validation scores
   - Test accuracy comparisons
   - Feature importance analysis

## 🚀 Streamlit Web Application

### Features
- Interactive prediction interface
- Real-time satisfaction probability
- Model performance insights
- Feature importance visualization
- Mobile-responsive design

### Application Sections
1. **Prediction Tab**
   - Passenger details form
   - Flight information input
   - Service rating sliders
   - Instant prediction results

2. **Insights Tab**
   - Model performance metrics
   - Key findings from analysis
   - Top influential features
   - Visual explanations

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/airline-satisfaction.git
cd airline-satisfaction

##Create and activate virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

##Install dependencies:
pip install -r requirements.txt

##Running the Notebook
jupyter notebook notebooks/Airline_Passenger_Analysis.ipynb

##Running the Streamlit App
streamlit run app/streamlit_app.py