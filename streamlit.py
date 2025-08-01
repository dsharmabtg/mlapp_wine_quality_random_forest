import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")


# Streamlit app
st.title("Wine Quality Predictor")
st.write("Enter the details of the parameters to predict the wine quality:")
st.write("Sample Data to test the Wine Quality")
data = [[8.1,0.380,0.28,2.1,0.066,13.0,30.0,0.9968,3.23,0.73,9.7,"Good"],[5.7,1.130,0.09,1.5,0.172,7.0,19.0,0.9940,3.50,0.48,9.8,"Bad"],[7.8,0.600,0.14,2.4,0.086,3.0,15.0,0.9975,3.42,0.60,10.8,"Average"]]

df = pd.DataFrame(data,columns=["Fixed acidity", "Volatile acidity", "Citric acid", "Residual sugar", "Chlorides", "Free sulfur dioxide", "Total sulfur dioxide", "Density", "pH", "Sulphates","Alcohol","Quality"])
df.round({"Fixed acidity":1, "Volatile acidity":2, "Citric acid":1, "Residual sugar":1, "Chlorides":3, "Free sulfur dioxide":1, "Total sulfur dioxide":1, "Density":6, "pH":2, "Sulphates":2,"Alcohol":1})
# Set the maximum column width to 100 characters
pd.set_option('display.max_colwidth', None)
st.table(df)

#st.write("Features:Fixed acidity; Volatile acidity; Citric acid; Residual sugar; chlorides; Free sulfur dioxide; Total sulfur dioxide; density; pH; sulphates")
#st.write("Good Quality:8.1;	0.380;	0.28;	2.1;	0.066;	13.0;	30.0;	0.9968;	3.23;	0.73;	9.7")
#st.write("Bad Quality:5.7;	1.130;	0.09;	1.5;	0.172;	7.0;	19.0;	0.9940;	3.50;	0.48;	9.8")
#st.write("Average Quality:7.8;	0.600;	0.14;	2.4;	0.086;	3.0;	15.0;	0.9975;	3.42;	0.60;	10.8")
# User input 
fixed_acidity = st.number_input("Fixed acidity:",min_value=4.6,max_value=15.9)
volatile_acidity = st.number_input("Volatile acidity:",min_value=0.12,max_value=1.58)
citric_acid = st.number_input("Citric acid:",min_value=0.0,max_value=1.0)
residual_sugar = st.number_input("Residual sugar:",min_value=0.9,max_value=15.5)
chlorides = st.number_input("Chlorides:",min_value=0.012,max_value=0.611)
free_sulfur_dioxide = st.number_input("Free sulfur dioxide:",min_value=1.0,max_value=72.0)
total_sulfur_dioxide = st.number_input("Total sulfur dioxide:",min_value=6.0,max_value=289.0)
density = st.number_input("Density:",min_value=0.990070,max_value=1.003690)
pH = st.number_input("pH:",min_value=2.74,max_value=4.01)
sulphates = st.number_input("Sulphates:",min_value=0.33,max_value=2.0)
alcohol = st.number_input("Alcohol:",min_value=8.4,max_value=14.9)
if st.button("Predict"):
    try:
        data = [
        [
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol
        ]
    ]
        scaled_data = scaler.transform(data)
        prediction = model.predict(scaled_data)
        st.write(f"Predicted Wine Quality: {prediction[0]}")
    except Exception as e:
        st.error(f"Error: {e}")
