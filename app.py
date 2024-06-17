
import streamlit as st
import joblib
import numpy as np

#load the model
model=joblib.load(model.pkl)

#Titleof the app
st.title("Loan offer acceptance predictor")

#Input fields
input_data= st.text_input("Enter input data : CSV ")

#Prediction Button
if st.button("Predict"):
    input_array=np.array([float(i) for i in input_data.split(',')]).reshape(1,-1)
    
    #Make prediction
    prediction=model.predict(input_array)
    
    #Display prediction
    st.write(f"Prediction: {prediction[0]}")


    
