from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib


# =========================================================
# LOAD MODEL
# =========================================================

MODEL_PATH = "best_layoffs_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully.")

except Exception as e:
    raise RuntimeError(
        f"Could not load model: {e}"
    )


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Layoffs Prediction API",
    description="Machine Learning API for Layoffs Prediction",
    version="1.0.0"
)


# =========================================================
# INPUT DATA SCHEMA
# =========================================================

class LayoffInput(BaseModel):

    record_id: str = "API-001"
    city: str = "New York"
    country: str = "United States"
    industry: str = "Retail"
    stage: str = "Series C"
    status: str = "Post-IPO"

    company_size_before: float = 500
    company_size_after: float = 300

    total_laid_off: float = 200
    funds_raised_mil: float = 100

    year: int = 2023

    # IMPORTANT: numeric because Experiment 4 trained it as numeric
    quarter: int = 1

    month_name: str = "January"
    day_of_week: str = "Monday"

    is_weekend: int = 0
    days_to_report: float = 10
    is_us: int = 1

    date_added: str = "2023-01-01"


# =========================================================
# HOME ENDPOINT
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Layoffs Prediction API is running",
        "predict_endpoint": "/predict",
        "documentation": "/docs",
        "health_check": "/health"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": True
    }


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(data: LayoffInput):

    try:

        # -------------------------------------------------
        # Convert request to DataFrame
        # -------------------------------------------------

        input_data = pd.DataFrame([{

            "record_id": data.record_id,

            "city": data.city,

            "country": data.country,

            "industry": data.industry,

            "stage": data.stage,

            "status": data.status,

            "company_size_before":
                data.company_size_before,

            "company_size_after":
                data.company_size_after,

            "total_laid_off":
                data.total_laid_off,

            "funds_raised_mil":
                data.funds_raised_mil,

            "year":
                data.year,

            "quarter":
                data.quarter,

            "month_name":
                data.month_name,

            "day_of_week":
                data.day_of_week,

            "is_weekend":
                data.is_weekend,

            "days_to_report":
                data.days_to_report,

            "is_us":
                data.is_us,

            "date_added":
                data.date_added
        }])


        print("\nInput received:")
        print(input_data)


        # -------------------------------------------------
        # MODEL PREDICTION
        # -------------------------------------------------

        prediction = model.predict(input_data)[0]


        # -------------------------------------------------
        # PREDICTION PROBABILITY
        # -------------------------------------------------

        probabilities = model.predict_proba(
            input_data
        )[0]


        # -------------------------------------------------
        # HUMAN READABLE RESULT
        # -------------------------------------------------

        if int(prediction) == 1:

            prediction_label = "50% or More"

        else:

            prediction_label = "Below 50%"


        # -------------------------------------------------
        # RETURN RESULT
        # -------------------------------------------------

        return {

            "prediction":
                int(prediction),

            "prediction_label":
                prediction_label,

            "probability_below_50":
                round(
                    float(probabilities[0]),
                    4
                ),

            "probability_50_or_more":
                round(
                    float(probabilities[1]),
                    4
                )
        }


    except Exception as e:

        print("Prediction error:", e)

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
