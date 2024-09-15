from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the saved model pipeline
model_pipeline = joblib.load('D:\Internship AI\API\housing_price_model_with_preprocessing.joblib')

class HousingData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str

def feature_engineering(data):
    data["rooms_per_household"] = data["total_rooms"] / data["households"]
    data["bedrooms_ratio"] = data["total_bedrooms"] / data["total_rooms"]
    data["people_per_household"] = data["population"] / data["households"]
    return data

@app.post("/predict/")
def predict(housing: HousingData):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([housing.dict()])

        # Apply feature engineering
        feature_engineered_data = feature_engineering(input_data)
        
        # Make prediction using the pipeline
        predicted_value = model_pipeline.predict(feature_engineered_data)[0]
        
        return {"predicted_house_value": predicted_value}
    
    except Exception as e:
        return {"error": str(e)}

# Root endpoint for testing
@app.get("/")
def read_root():
    return {"message": "Welcome to the Housing Price Prediction API!"}
