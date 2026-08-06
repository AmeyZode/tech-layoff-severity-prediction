# Tech Layoff Severity Prediction

An end-to-end Machine Learning project that predicts workforce layoff severity using historical global technology layoff data. The project demonstrates the complete Machine Learning lifecycle—from dataset preparation and feature engineering to model deployment and dashboard visualization—using open-source tools.

---

## Project Overview

Mass layoffs have become increasingly common across the technology sector due to economic slowdowns, funding shortages, organizational restructuring, and changing market conditions. Historical layoff records contain valuable information that can help identify workforce reduction patterns and predict future layoff severity.

This project uses a merged global technology layoffs dataset containing over **7,400 real-world layoff records** collected from multiple public sources.

The project follows an eight-experiment workflow covering the complete MLOps pipeline.

---

## Problem Statement

Develop a machine learning model capable of predicting layoff severity based on company characteristics such as:

- Industry
- Company Size
- Funding Stage
- Funds Raised
- Country
- Geographic Region
- Historical Trends

The project also explains model predictions using Explainable AI techniques and deploys the trained model as a production-ready API.

---

## Dataset

**Dataset:** Merged Global Technology Layoffs Dataset

### Dataset Statistics

- Records: **7,441**
- Features: **22**
- Format: CSV

### Main Features

- Company
- Industry
- Country
- Region
- Company Size Before Layoff
- Company Size After Layoff
- Funding Stage
- Funds Raised
- Total Employees Laid Off
- Percentage Laid Off
- Year
- Latitude & Longitude

---

## Project Workflow

### Experiment 1
- Case Study Framing
- Dataset Preparation
- Git & DVC Versioning

### Experiment 2
- Data Profiling
- Missing Value Handling
- Feature Engineering

### Experiment 3
- Exploratory Data Analysis
- Statistical Analysis

### Experiment 4
- Machine Learning
- Experiment Tracking

### Experiment 5
- Explainable AI
- Fairness Evaluation

### Experiment 6
- FastAPI Deployment
- Docker Containerization

### Experiment 7
- CI/CD Pipeline
- GitHub Actions

### Experiment 8
- Streamlit Dashboard
- Responsible AI Report
- Final Portfolio

---

## Technologies Used

### Programming

- Python

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn
- Plotly

### Machine Learning

- Scikit-learn
- XGBoost
- LightGBM

### Explainable AI

- SHAP

### Deployment

- FastAPI
- Docker

### MLOps

- Git
- GitHub
- DVC
- MLflow
- GitHub Actions

### Dashboard

- Streamlit

---

## Project Structure

```
Tech-Layoff-Severity-Prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│
├── models/
│
├── api/
│
├── dashboard/
│
├── reports/
│
├── experiments/
│
├── .dvc/
│
├── requirements.txt
├── README.md
├── dvc.yaml
└── .gitignore
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/<your-username>/ADS.git
```

Navigate into the project

```bash
cd ADS
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## DVC Commands

Initialize DVC

```bash
dvc init
```

Track dataset

```bash
dvc add data/raw/merged_layoffs-2.csv
```

Pull dataset

```bash
dvc pull
```

Push dataset

```bash
dvc push
```

---

## Expected Outputs

- Cleaned Dataset
- Feature Engineered Dataset
- Trained Machine Learning Models
- Model Performance Reports
- SHAP Explainability Visualizations
- FastAPI REST API
- Docker Image
- CI/CD Workflow
- Streamlit Dashboard

---

## Future Improvements

- Real-time layoff prediction
- Automated data updates
- Time-series forecasting
- Deep Learning models
- Cloud deployment
- Interactive business intelligence dashboard

---

## License

This project is developed for academic and educational purposes.

---

## Author

**Amey Zode**

Bachelor of Engineering (Computer Engineering - Artificial Intelligence)
