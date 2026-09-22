from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert "predict_endpoint" in data


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_prediction():

    payload = {
        "record_id": "CI-001",
        "city": "New York",
        "country": "United States",
        "industry": "Retail",
        "stage": "Series C",
        "status": "Post-IPO",

        "company_size_before": 500,
        "company_size_after": 300,

        "total_laid_off": 200,
        "funds_raised_mil": 100,

        "year": 2023,
        "quarter": 1,
        "month_name": "January",
        "day_of_week": "Monday",

        "is_weekend": 0,
        "days_to_report": 10,
        "is_us": 1,

        "date_added": "2023-01-01"
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "prediction_label" in data
    assert "probability_below_50" in data
    assert "probability_50_or_more" in data

    assert data["prediction"] in [0, 1]

    assert 0 <= data["probability_below_50"] <= 1
    assert 0 <= data["probability_50_or_more"] <= 1
