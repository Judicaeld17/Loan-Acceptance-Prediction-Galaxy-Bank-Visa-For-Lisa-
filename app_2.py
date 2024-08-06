import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('model.pkl')

# Title of the app
st.title("Loan Offer Acceptance Predictor")

# File uploader for CSV
uploaded_file = st.file_uploader("Upload CSV file for predictions", type=["csv"])

if uploaded_file is not None:
    # Debug message to check if the file is being read
    st.write("File uploaded successfully!")

    # Read the uploaded file into a DataFrame
    try:
        data = pd.read_csv(uploaded_file)

        # Display the uploaded data
        st.write("Uploaded Data:")
        st.write(data)

        # Check if the expected columns are present
        expected_columns = ['Age', 'Experience', 'Income', 'ZIP_Code', 'Family', 'CCAvg',
                            'Education', 'Mortgage', 'Securities_Account', 'CD_Account',
                            'Online', 'CreditCard']
        
        if all(column in data.columns for column in expected_columns):
            # Preprocess the data
            data['Securities_Account'] = data['Securities_Account'].apply(lambda x: 1 if x == "Yes" else 0)
            data['CD_Account'] = data['CD_Account'].apply(lambda x: 1 if x == "Yes" else 0)
            data['Online'] = data['Online'].apply(lambda x: 1 if x == "Yes" else 0)
            data['CreditCard'] = data['CreditCard'].apply(lambda x: 1 if x == "Yes" else 0)

            # Extract the features from the DataFrame
            features = data[expected_columns]

            # Make predictions for each row in the DataFrame
            predictions = model.predict(features)

            # Add predictions to the DataFrame
            data['Prediction'] = predictions

            # Display the predictions
            st.write("Predictions:")
            st.write(data)
        else:
            st.write("Error: The uploaded CSV does not contain all the required columns.")
    except Exception as e:
        st.write(f"Error reading the uploaded file: {e}")
else:
    st.write("Please upload a CSV file.")
