import streamlit as st
import requests
import sys

from src.exception import CustomException
from src.logger import logging

logging.info("Streamlit app started")

st.set_page_config(page_title="Student Performance Predictor", layout="centered")
st.title("Student Performance Predictor")

# ======================
# Inputs
# ======================
gender = st.selectbox("Gender", ["male", "female"])
race_ethnicity = st.selectbox("Race / Ethnicity", ["group A", "group B", "group C"])
parental_level_of_education = st.selectbox(
    "Parental Level of Education",
    [
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree",
    ],
)
lunch = st.selectbox("Lunch", ["free/reduced", "standard"])
test_preparation_course = st.selectbox(
    "Test Preparation Course", ["completed", "not completed"]
)

reading_score = st.number_input("Reading Score", 0, 100)
writing_score = st.number_input("Writing Score", 0, 100)

# ======================
# Predict
# ======================
if st.button("Predict Performance"):
    logging.info("Sending prediction request to FastAPI server")
    payload = {
        "gender": gender,
        "race_ethnicity": race_ethnicity,
        "parental_level_of_education": parental_level_of_education,
        "lunch": lunch,
        "test_preparation_course": test_preparation_course,
        "reading_score": reading_score,
        "writing_score": writing_score,
    }

    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if res.status_code == 200:
            st.success(f"Predicted Math Score: {res.json()['prediction']}")
        else:
            st.error("Backend error")

    except Exception as e:
        logging.info("Error connecting to FastAPI server")
        st.error("FastAPI server not running")
        raise CustomException(e, sys)
