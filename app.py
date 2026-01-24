import sys
from fastapi import FastAPI
from pydantic import BaseModel

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging

app = FastAPI()


class StudentInput(BaseModel):
    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: int
    writing_score: int


@app.get("/")
def index():
    return {"message": "Student Performance Predictor API running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: StudentInput):
    try:
        logging.info("Prediction request received")

        custom_data = CustomData(
            gender=data.gender,
            race_ethnicity=data.race_ethnicity,
            parental_level_of_education=data.parental_level_of_education,
            lunch=data.lunch,
            test_preparation_course=data.test_preparation_course,
            reading_score=data.reading_score,
            writing_score=data.writing_score,
        )

        df = custom_data.get_data_as_data_frame()

        pipeline = PredictPipeline()
        result = pipeline.predict(df)

        return {"prediction": int(result[0])}

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise CustomException(e, sys)
