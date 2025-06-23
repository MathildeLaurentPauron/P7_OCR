# streamlit_app.py
import streamlit as st
import requests
import json

# Streamlit UI
st.title("ML Prediction Interface")
st.write("This Streamlit app communicates with a Flask API to get machine learning predictions.")

# Input Fields
st.header("Input Question")
question = st.text_input("Enter your question:", "")

# API URL
api_url = "http://127.0.0.1:8999/predict_tags"
# Predict Button
if st.button("Predict"):
    try:
        # Append question as a query parameter in the URL
        full_url = f"{api_url}/{question}"

        # Make GET request to the API
        response = requests.get(full_url)

        # Parse and display response
        if response.status_code == 200:
            prediction = response.json()
            st.success(f"Prediction: {prediction}")
        else:
            st.error(f"Error {response.status_code}: {response.text}")

    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")