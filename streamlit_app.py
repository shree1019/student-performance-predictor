import streamlit as st
import requests


API_URL = "http://127.0.0.1:5000/predict"


st.title("Student Performance Predictor")
st.markdown("This app predicts a student's future grade and trend based on their past performance.")


g1 = st.number_input("Test 1 Marks:", min_value=0, max_value=20, value=10, step=1)
g2 = st.number_input("Test 2 Marks:", min_value=0, max_value=20, value=10, step=1)
studytime = st.number_input("Study Time:", min_value=1, max_value=4, value=2, step=1)
failures = st.number_input("Failed tests:", min_value=0, max_value=5, value=0, step=1)
absences = st.number_input("Absences :", min_value=0, value=0, step=1)


if st.button("Predict"):
    
    input_data = {
        "G1": g1,
        "G2": g2,
        "studytime": studytime,
        "failures": failures,
        "absences": absences
    }
    
    
    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            prediction = response.json()
            predicted_grade = prediction.get("predicted_grade", "N/A")
            trend = prediction.get("trend", "N/A")
            
            
            st.success(f"Predicted Grade: {predicted_grade}")
            st.info(f"Trend: {trend}")
        else:
            st.error(f"Error: {response.json().get('error', 'Unknown error')}")
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
