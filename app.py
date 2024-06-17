import streamlit as st
import joblib
import numpy as np

#load the model
model=joblib.load('model.pkl')

#Titleof the app
st.title("Loan offer acceptance predictor")

#Prediction Button

# ID = st.number_input("ID", value=2)
Age = st.number_input("Age", value=45)
Experience = st.number_input("Experience (years)", value=19)
Income = st.number_input("Income ($)", value=34000)
ZIP_Code = st.number_input("ZIP Code", value=90089)
Family = st.number_input("Family (members)", value=3)
CCAvg = st.number_input("Average Credit Card Spending ($)", value=1500)
Education = st.selectbox("Education Level", options=[1, 2, 3], index=0)
Mortgage = st.number_input("Mortgage ($)", value=0)
Securities_Account = st.selectbox("Securities Account", options=["Yes", "No"], index=0)
CD_Account = st.selectbox("CD Account", options=["Yes", "No"], index=1)
Online = st.selectbox("Online", options=["Yes", "No"], index=1)
CreditCard = st.selectbox("Credit Card", options=["Yes", "No"], index=1)

# Button to make predictions
if st.button("Predict"):
    # Convert input data to numpy array
    input_array = np.array([Age, Experience, Income, ZIP_Code, Family, CCAvg, 
                            Education, Mortgage, 
                            1 if Securities_Account == "Yes" else 0,
                            1 if CD_Account == "Yes" else 0,
                            1 if Online == "Yes" else 0,
                            1 if CreditCard == "Yes" else 0]).reshape(1, -1)
    
    #Make prediction
    prediction=model.predict(input_array)
    
    #Display prediction
    st.write(f"Prediction: {prediction[0]}")

    
    
