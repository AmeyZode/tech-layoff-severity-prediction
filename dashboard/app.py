import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Tech Layoff Severity Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "layoffs-api",
    "best_layoffs_model.pkl",
)

DEFAULT_DATA_PATHS = [
    os.path.join(BASE_DIR, "data", "cleaned_layoffs.csv"),
    os.path.join(BASE_DIR, "data", "cleaned_dataset.csv"),
    os.path.join(BASE_DIR, "data", "layoffs_cleaned.csv"),
    os.path.join(BASE_DIR, "cleaned_layoffs.csv"),
]


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None, (
            "Model file not found. Place "
            "`best_layoffs_model.pkl` inside "
            "`layoffs-api/`."
        )

    try:

        model = joblib.load(MODEL_PATH)

        return model, None

    except Exception as exc:

        return None, str(exc)


model, model_error = load_model()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

MODEL_COLUMNS = [
    "record_id",
    "city",
    "country",
    "industry",
    "stage",
    "status",
    "company_size_before",
    "company_size_after",
    "total_laid_off",
    "funds_raised_mil",
    "year",
    "quarter",
    "month_name",
    "day_of_week",
    "is_weekend",
    "days_to_report",
    "is_us",
    "date_added",
]


def normalize_columns(df):

    """
    Convert common original Layoffs dataset column names
    into the feature names used by the trained API model.
    """

    rename_map = {
        "Record ID": "record_id",
        "RecordID": "record_id",
        "Company": "company",
        "City": "city",
        "Country": "country",
        "Industry": "industry",
        "Stage": "stage",
        "Company Size (Before)": "company_size_before",
        "Company Size Before": "company_size_before",
        "Company Size (After)": "company_size_after",
        "Company Size After": "company_size_after",
        "Total Laid Off": "total_laid_off",
        "% Laid Off": "percent_laid_off",
        "Funds Raised ($M)": "funds_raised_mil",
        "Funds Raised": "funds_raised_mil",
        "Date": "date",
        "Date Added": "date_added",
        "Status": "status",
        "Status ": "status",
    }

    result = df.copy()

    result.columns = [
        str(col).strip()
        for col in result.columns
    ]

    result = result.rename(
        columns=rename_map
    )

    return result


def create_model_features(df):

    """
    Prepare a dataframe for the exact model schema.
    """

    data = normalize_columns(df)

    # -----------------------------------------------------
    # Date processing
    # -----------------------------------------------------

    if "date" in data.columns:

        parsed_date = pd.to_datetime(
            data["date"],
            errors="coerce"
        )

        if "year" not in data.columns:
            data["year"] = parsed_date.dt.year

        if "quarter" not in data.columns:
            data["quarter"] = parsed_date.dt.quarter

        if "month_name" not in data.columns:
            data["month_name"] = parsed_date.dt.month_name()

        if "day_of_week" not in data.columns:
            data["day_of_week"] = parsed_date.dt.day_name()

        if "is_weekend" not in data.columns:
            data["is_weekend"] = (
                parsed_date.dt.dayofweek >= 5
            ).astype(int)

    # -----------------------------------------------------
    # Date Added
    # -----------------------------------------------------

    if "date_added" in data.columns:

        data["date_added"] = (
            pd.to_datetime(
                data["date_added"],
                errors="coerce"
            )
            .astype(str)
        )

    # -----------------------------------------------------
    # Numeric conversions
    # -----------------------------------------------------

    numeric_columns = [
        "company_size_before",
        "company_size_after",
        "total_laid_off",
        "funds_raised_mil",
        "year",
        "quarter",
        "is_weekend",
        "days_to_report",
        "is_us",
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = pd.to_numeric(
                data[column],
                errors="coerce"
            )

    # -----------------------------------------------------
    # Create missing geographic flag
    # -----------------------------------------------------

    if "is_us" not in data.columns:

        if "country" in data.columns:

            data["is_us"] = (
                data["country"]
                .astype(str)
                .str.lower()
                .isin([
                    "united states",
                    "usa",
                    "us",
                ])
                .astype(int)
            )

    # -----------------------------------------------------
    # Days to report
    # -----------------------------------------------------

    if "days_to_report" not in data.columns:

        data["days_to_report"] = 0

    # -----------------------------------------------------
    # Ensure all expected columns exist
    # -----------------------------------------------------

    for column in MODEL_COLUMNS:

        if column not in data.columns:

            data[column] = np.nan

    return data[MODEL_COLUMNS]


def target_from_dataset(df):

    """
    Create the binary target:
    0 = below 50%
    1 = 50% or more
    """

    data = normalize_columns(df)

    if "percent_laid_off" not in data.columns:

        return None

    percentage = pd.to_numeric(
        data["percent_laid_off"],
        errors="coerce"
    )

    return (
        percentage >= 50
    ).astype(int)


def probability_prediction(model, features):

    prediction = model.predict(features)

    probabilities = model.predict_proba(
        features
    )

    return prediction, probabilities


def calculate_drift(reference, current):

    """
    Simple numerical distribution drift using
    standardized mean difference.
    """

    rows = []

    numeric_columns = list(
        set(reference.select_dtypes(
            include=np.number
        ).columns)
        &
        set(current.select_dtypes(
            include=np.number
        ).columns)
    )

    for column in numeric_columns:

        ref = pd.to_numeric(
            reference[column],
            errors="coerce"
        ).dropna()

        cur = pd.to_numeric(
            current[column],
            errors="coerce"
        ).dropna()

        if len(ref) < 2 or len(cur) < 2:
            continue

        pooled_std = np.sqrt(
            (
                ref.var() +
                cur.var()
            ) / 2
        )

        if pooled_std == 0:
            drift = 0
        else:
            drift = abs(
                ref.mean() - cur.mean()
            ) / pooled_std

        if drift < 0.1:
            status = "Low"

        elif drift < 0.3:
            status = "Moderate"

        else:
            status = "High"

        rows.append({
            "Feature": column,
            "Standardized Mean Difference":
                round(drift, 4),
            "Drift": status,
        })

    return pd.DataFrame(rows)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("📊 Layoff Severity")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Overview",
        "🔮 Prediction",
        "📈 Model Performance",
        "🔍 SHAP Explainability",
        "📉 Data Drift",
        "⚖️ Fairness",
        "🛡 Responsible AI",
        "📋 Dataset",
    ],
)


# =========================================================
# MODEL STATUS
# =========================================================

if model is None:

    st.sidebar.error(
        "Model not loaded"
    )

else:

    st.sidebar.success(
        "Model loaded"
    )

    st.sidebar.caption(
        "scikit-learn compatible model"
    )


# =========================================================
# OVERVIEW
# =========================================================

if page == "🏠 Overview":

    st.title(
        "Tech Layoff Severity Prediction"
    )

    st.subheader(
        "End-to-End Machine Learning Dashboard"
    )

    st.markdown(
        """
        This application provides an interactive interface
        for the Tech Layoff Severity Prediction project.

        The project covers the complete machine learning
        lifecycle from data profiling and preprocessing to
        model training, explainability, deployment,
        monitoring and responsible AI.
        """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Project",
            "Layoff Severity"
        )

    with col2:
        st.metric(
            "Prediction Classes",
            "2"
        )

    with col3:
        st.metric(
            "Explainability",
            "SHAP + LIME"
        )

    with col4:
        st.metric(
            "Deployment",
            "FastAPI + Docker"
        )

    st.divider()

    st.subheader(
        "Project Pipeline"
    )

    st.code(
        """
Data Profiling
      ↓
Data Cleaning
      ↓
Feature Engineering
      ↓
EDA & Statistical Analysis
      ↓
ML Modeling
      ↓
MLflow Experiment Tracking
      ↓
SHAP + LIME + Fairness
      ↓
FastAPI
      ↓
Docker
      ↓
GitHub Actions CI/CD
      ↓
Streamlit Dashboard
        """,
        language="text",
    )

    st.info(
        "Use the sidebar to explore predictions, "
        "model performance, explainability, drift "
        "and responsible AI information."
    )


# =========================================================
# PREDICTION
# =========================================================

elif page == "🔮 Prediction":

    st.title(
        "🔮 Layoff Severity Prediction"
    )

    if model is None:

        st.error(
            model_error
        )

        st.stop()

    st.write(
        "Enter company and layoff information."
    )

    col1, col2 = st.columns(2)

    with col1:

        record_id = st.text_input(
            "Record ID",
            "DASH-001"
        )

        city = st.text_input(
            "City",
            "New York"
        )

        country = st.text_input(
            "Country",
            "United States"
        )

        industry = st.selectbox(
            "Industry",
            [
                "Technology",
                "Retail",
                "Finance",
                "Healthcare",
                "Transportation",
                "Food",
                "Education",
                "Other",
            ]
        )

        stage = st.selectbox(
            "Stage",
            [
                "Seed",
                "Series A",
                "Series B",
                "Series C",
                "Series D",
                "Series E",
                "Series F",
                "Post-IPO",
            ]
        )

        status = st.selectbox(
            "Status",
            [
                "Private",
                "Post-IPO",
            ]
        )

    with col2:

        company_size_before = st.number_input(
            "Company Size Before",
            min_value=0.0,
            value=500.0,
        )

        company_size_after = st.number_input(
            "Company Size After",
            min_value=0.0,
            value=300.0,
        )

        total_laid_off = st.number_input(
            "Total Laid Off",
            min_value=0.0,
            value=200.0,
        )

        funds_raised_mil = st.number_input(
            "Funds Raised ($M)",
            min_value=0.0,
            value=100.0,
        )

        year = st.number_input(
            "Year",
            min_value=2000,
            max_value=2100,
            value=2023,
        )

        quarter = st.selectbox(
            "Quarter",
            [1, 2, 3, 4],
        )

        month_name = st.selectbox(
            "Month",
            [
                "January",
                "February",
                "March",
                "April",
                "May",
                "June",
                "July",
                "August",
                "September",
                "October",
                "November",
                "December",
            ]
        )

        day_of_week = st.selectbox(
            "Day",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        )

        days_to_report = st.number_input(
            "Days to Report",
            min_value=0.0,
            value=10.0,
        )

        date_added = st.text_input(
            "Date Added",
            "2023-01-01",
        )

    is_weekend = int(
        day_of_week in [
            "Saturday",
            "Sunday",
        ]
    )

    is_us = int(
        country.lower()
        in [
            "united states",
            "usa",
            "us",
        ]
    )

    st.divider()

    predict_button = st.button(
        "🚀 Predict Layoff Severity",
        type="primary",
        use_container_width=True,
    )

    if predict_button:

        input_data = pd.DataFrame([{

            "record_id": record_id,

            "city": city,

            "country": country,

            "industry": industry,

            "stage": stage,

            "status": status,

            "company_size_before":
                company_size_before,

            "company_size_after":
                company_size_after,

            "total_laid_off":
                total_laid_off,

            "funds_raised_mil":
                funds_raised_mil,

            "year":
                year,

            "quarter":
                quarter,

            "month_name":
                month_name,

            "day_of_week":
                day_of_week,

            "is_weekend":
                is_weekend,

            "days_to_report":
                days_to_report,

            "is_us":
                is_us,

            "date_added":
                date_added,
        }])

        try:

            prediction, probabilities = (
                probability_prediction(
                    model,
                    input_data
                )
            )

            prediction_value = int(
                prediction[0]
            )

            below_probability = float(
                probabilities[0][0]
            )

            high_probability = float(
                probabilities[0][1]
            )

            if prediction_value == 1:

                label = "50% or More"

            else:

                label = "Below 50%"

            st.success(
                f"Prediction: {label}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Prediction",
                    label
                )

            with col2:

                st.metric(
                    "Below 50%",
                    f"{below_probability * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "50% or More",
                    f"{high_probability * 100:.2f}%"
                )

            chart_data = pd.DataFrame(
                {
                    "Class": [
                        "Below 50%",
                        "50% or More",
                    ],

                    "Probability": [
                        below_probability,
                        high_probability,
                    ],
                }
            )

            st.subheader(
                "Prediction Probabilities"
            )

            st.bar_chart(
                chart_data.set_index(
                    "Class"
                )
            )

            with st.expander(
                "View Input Features"
            ):

                st.dataframe(
                    input_data,
                    use_container_width=True
                )

        except Exception as exc:

            st.error(
                f"Prediction failed: {exc}"
            )


# =========================================================
# DATASET UPLOAD
# =========================================================

elif page == "📋 Dataset":

    st.title(
        "📋 Dataset Explorer"
    )

    st.write(
        """
        Upload the cleaned Layoffs dataset to inspect
        its structure, statistics and target distribution.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload cleaned Layoffs CSV",
        type=["csv"],
    )

    df = None

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )

        except Exception as exc:

            st.error(
                f"Could not read file: {exc}"
            )

    else:

        for path in DEFAULT_DATA_PATHS:

            if os.path.exists(path):

                try:

                    df = pd.read_csv(path)

                    st.info(
                        f"Loaded: {path}"
                    )

                    break

                except Exception:
                    pass

    if df is None:

        st.warning(
            "Upload the cleaned dataset to "
            "view dataset analysis."
        )

        st.stop()

    df = normalize_columns(df)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            f"{df.isna().sum().sum():,}"
        )

    with col4:
        st.metric(
            "Duplicate Rows",
            f"{df.duplicated().sum():,}"
        )

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    st.subheader(
        "Numerical Summary"
    )

    st.dataframe(
        df.describe(
            include="all"
        ).T,
        use_container_width=True
    )

    if "percent_laid_off" in df.columns:

        target = target_from_dataset(df)

        if target is not None:

            st.subheader(
                "Layoff Severity Distribution"
            )

            counts = target.value_counts()

            labels = [
                "Below 50%",
                "50% or More",
            ]

            chart_df = pd.DataFrame(
                {
                    "Class": labels,
                    "Count": [
                        counts.get(0, 0),
                        counts.get(1, 0),
                    ],
                }
            )

            st.bar_chart(
                chart_df.set_index(
                    "Class"
                )
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "📈 Model Performance":

    st.title(
        "📈 Model Performance"
    )

    if model is None:

        st.error(
            model_error
        )

        st.stop()

    uploaded_file = st.file_uploader(
        "Upload evaluation dataset",
        type=["csv"],
        key="metrics_upload",
    )

    df = None

    if uploaded_file is not None:

        df = pd.read_csv(
            uploaded_file
        )

    else:

        for path in DEFAULT_DATA_PATHS:

            if os.path.exists(path):

                try:

                    df = pd.read_csv(path)
                    break

                except Exception:
                    pass

    if df is None:

        st.info(
            "Upload the cleaned dataset containing "
            "`% Laid Off` to calculate model metrics."
        )

        st.stop()

    df = normalize_columns(df)

    y_true = target_from_dataset(df)

    if y_true is None:

        st.warning(
            "The dataset does not contain `% Laid Off`, "
            "so actual classification metrics cannot "
            "be calculated."
        )

        st.stop()

    features = create_model_features(
        df
    )

    valid_rows = (
        y_true.notna()
    )

    features = features.loc[
        valid_rows
    ]

    y_true = y_true.loc[
        valid_rows
    ]

    try:

        predictions, probabilities = (
            probability_prediction(
                model,
                features
            )
        )

        predictions = np.asarray(
            predictions
        )

        probability_positive = (
            probabilities[:, 1]
        )

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:
            st.metric(
                "Accuracy",
                f"{accuracy:.2%}"
            )

        with col2:
            st.metric(
                "Precision",
                f"{precision:.2%}"
            )

        with col3:
            st.metric(
                "Recall",
                f"{recall:.2%}"
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{f1:.2%}"
            )

        try:

            auc = roc_auc_score(
                y_true,
                probability_positive
            )

            st.metric(
                "ROC-AUC",
                f"{auc:.2%}"
            )

        except ValueError:

            auc = None

        st.subheader(
            "Confusion Matrix"
        )

        matrix = confusion_matrix(
            y_true,
            predictions
        )

        fig, ax = plt.subplots()

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "Below 50%",
                "50% or More",
            ],
            yticklabels=[
                "Below 50%",
                "50% or More",
            ],
            ax=ax,
        )

        ax.set_xlabel(
            "Predicted"
        )

        ax.set_ylabel(
            "Actual"
        )

        st.pyplot(
            fig
        )

        plt.close(fig)

    except Exception as exc:

        st.error(
            f"Unable to calculate metrics: {exc}"
        )


# =========================================================
# SHAP
# =========================================================
elif page == "🔍 SHAP Explainability":

    st.title("🔍 SHAP Explainability")

    st.write(
        """
        SHAP explains how the transformed model features
        contribute to the prediction. Because the trained
        model contains preprocessing for categorical and
        numerical variables, SHAP is applied after the
        preprocessing stage.
        """
    )

    if model is None:

        st.error(model_error)
        st.stop()

    uploaded_file = st.file_uploader(
        "Upload cleaned dataset for SHAP analysis",
        type=["csv"],
        key="shap_upload",
    )

    if uploaded_file is None:

        st.info(
            "Upload the cleaned dataset to generate "
            "SHAP explanations."
        )

        st.stop()

    try:

        # -------------------------------------------------
        # Load dataset
        # -------------------------------------------------

        df = pd.read_csv(
            uploaded_file
        )

        df = normalize_columns(df)

        features = create_model_features(
            df
        )

        if len(features) == 0:

            st.warning(
                "The uploaded dataset contains no rows."
            )

            st.stop()

        # -------------------------------------------------
        # Sample data
        # -------------------------------------------------

        sample_size = min(
            100,
            len(features)
        )

        sample = features.sample(
            sample_size,
            random_state=42
        ).reset_index(drop=True)

        st.info(
            f"Using {sample_size} samples for SHAP analysis."
        )

        if not st.button(
            "Generate SHAP Explanation",
            type="primary",
        ):

            st.stop()

        # -------------------------------------------------
        # Check whether model is a Pipeline
        # -------------------------------------------------

        if not hasattr(model, "steps"):

            st.error(
                """
                The saved model is not an sklearn Pipeline.

                The SHAP implementation below expects the
                preprocessing and estimator to be stored
                together in the trained Pipeline.
                """
            )

            st.stop()

        # -------------------------------------------------
        # Extract preprocessing + final estimator
        # -------------------------------------------------

        preprocessing = model[:-1]

        estimator = model.steps[-1][1]

        st.write(
            f"Final estimator: `{type(estimator).__name__}`"
        )

        # -------------------------------------------------
        # Transform raw features
        # -------------------------------------------------

        transformed = preprocessing.transform(
            sample
        )

        # Convert sparse matrix to dense
        if hasattr(
            transformed,
            "toarray"
        ):

            transformed = transformed.toarray()

        transformed = np.asarray(
            transformed
        )

        # -------------------------------------------------
        # Obtain transformed feature names
        # -------------------------------------------------

        try:

            feature_names = (
                preprocessing
                .get_feature_names_out()
            )

        except Exception:

            feature_names = np.array(
                [
                    f"Feature_{i}"
                    for i in range(
                        transformed.shape[1]
                    )
                ]
            )

        feature_names = np.asarray(
            feature_names
        )

        # -------------------------------------------------
        # Create transformed dataframe
        # -------------------------------------------------

        transformed_df = pd.DataFrame(
            transformed,
            columns=feature_names
        )

        st.write(
            f"Transformed feature matrix: "
            f"`{transformed_df.shape[0]} × "
            f"{transformed_df.shape[1]}`"
        )

        # -------------------------------------------------
        # Background data
        # -------------------------------------------------

        background_size = min(
            20,
            len(transformed_df)
        )

        background = transformed_df.iloc[
            :background_size
        ]

        # -------------------------------------------------
        # Prediction function
        # -------------------------------------------------

        def predict_transformed(X):

            X = np.asarray(X)

            return estimator.predict_proba(
                X
            )[:, 1]

        # -------------------------------------------------
        # SHAP explainer
        # -------------------------------------------------

        with st.spinner(
            "Calculating SHAP values..."
        ):

            explainer = shap.Explainer(
                predict_transformed,
                background,
                feature_names=feature_names,
            )

            shap_values = explainer(
                transformed_df
            )

        st.success(
            "SHAP analysis completed successfully."
        )

        # =================================================
        # GLOBAL FEATURE IMPORTANCE
        # =================================================

        st.subheader(
            "Global Feature Importance"
        )

        mean_abs_shap = np.abs(
            shap_values.values
        ).mean(
            axis=0
        )

        importance_df = pd.DataFrame(
            {
                "Feature":
                    feature_names,

                "Mean |SHAP|":
                    mean_abs_shap,
            }
        ).sort_values(
            "Mean |SHAP|",
            ascending=False,
        )

        importance_df = (
            importance_df
            .head(20)
        )

        st.dataframe(
            importance_df,
            use_container_width=True,
        )

        # -------------------------------------------------
        # Importance chart
        # -------------------------------------------------

        chart_df = (
            importance_df
            .sort_values(
                "Mean |SHAP|"
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        ax.barh(
            chart_df["Feature"],
            chart_df["Mean |SHAP|"],
        )

        ax.set_xlabel(
            "Mean Absolute SHAP Value"
        )

        ax.set_ylabel(
            "Feature"
        )

        ax.set_title(
            "Global SHAP Feature Importance"
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

        # =================================================
        # SHAP SUMMARY PLOT
        # =================================================

        st.subheader(
            "SHAP Summary Plot"
        )

        fig = plt.figure(
            figsize=(10, 7)
        )

        shap.summary_plot(
            shap_values.values,
            transformed_df,
            feature_names=feature_names,
            show=False,
            max_display=20,
        )

        plt.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True,
        )

        plt.close(fig)

        # =================================================
        # LOCAL EXPLANATION
        # =================================================

        st.subheader(
            "Local Explanation"
        )

        selected_index = st.slider(
            "Select sample",
            min_value=0,
            max_value=len(sample) - 1,
            value=0,
        )

        local_values = (
            shap_values
            .values[selected_index]
        )

        local_df = pd.DataFrame(
            {
                "Feature":
                    feature_names,

                "SHAP Value":
                    local_values,

                "Absolute Impact":
                    np.abs(local_values),
            }
        ).sort_values(
            "Absolute Impact",
            ascending=False,
        )

        st.dataframe(
            local_df.head(15),
            use_container_width=True,
        )

        # -------------------------------------------------
        # Local bar chart
        # -------------------------------------------------

        local_chart = (
            local_df
            .head(10)
            .sort_values(
                "SHAP Value"
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        ax.barh(
            local_chart["Feature"],
            local_chart["SHAP Value"],
        )

        ax.axvline(
            0,
            linewidth=1,
        )

        ax.set_xlabel(
            "SHAP Value"
        )

        ax.set_title(
            "Local Feature Contributions"
        )

        plt.tight_layout()

        st.pyplot(
            fig
        )

        plt.close(fig)

        # =================================================
        # INTERPRETATION
        # =================================================

        st.subheader(
            "How to Interpret the Explanation"
        )

        st.markdown(
            """
            **Mean |SHAP|**

            Measures the average magnitude of a feature's
            contribution across the selected observations.

            **Positive SHAP value**

            Pushes the prediction toward the positive class
            (`50% or More`).

            **Negative SHAP value**

            Pushes the prediction toward the negative class
            (`Below 50%`).

            **Important note**

            SHAP describes the model's learned associations.
            It does not establish a causal relationship between
            a feature and layoff severity.
            """
        )

    except Exception as exc:

        st.error(
            f"SHAP analysis failed: {exc}"
        )

        st.exception(
            exc
        )
# =========================================================
# DATA DRIFT
# =========================================================

elif page == "📉 Data Drift":

    st.title(
        "📉 Data Drift Monitoring"
    )

    st.write(
        """
        Data drift compares a reference dataset with a
        newer/current dataset to identify changes in
        numerical feature distributions.
        """
    )

    reference_file = st.file_uploader(
        "Upload reference dataset",
        type=["csv"],
        key="reference_drift",
    )

    current_file = st.file_uploader(
        "Upload current dataset",
        type=["csv"],
        key="current_drift",
    )

    if (
        reference_file is None
        or current_file is None
    ):

        st.info(
            """
            Upload both datasets to perform the drift check.

            Reference dataset:
            training or historical data.

            Current dataset:
            newer data or a new batch of observations.
            """
        )

        st.stop()

    reference = pd.read_csv(
        reference_file
    )

    current = pd.read_csv(
        current_file
    )

    reference = normalize_columns(
        reference
    )

    current = normalize_columns(
        current
    )

    results = calculate_drift(
        reference,
        current
    )

    if results.empty:

        st.warning(
            "No comparable numerical features "
            "were found."
        )

    else:

        st.subheader(
            "Drift Results"
        )

        st.dataframe(
            results,
            use_container_width=True
        )

        high_drift = (
            results["Drift"]
            == "High"
        ).sum()

        moderate_drift = (
            results["Drift"]
            == "Moderate"
        ).sum()

        low_drift = (
            results["Drift"]
            == "Low"
        ).sum()

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Low Drift",
                low_drift
            )

        with col2:

            st.metric(
                "Moderate Drift",
                moderate_drift
            )

        with col3:

            st.metric(
                "High Drift",
                high_drift
            )

        if high_drift > 0:

            st.warning(
                "High distribution drift detected "
                "in one or more numerical features."
            )

        else:

            st.success(
                "No high numerical distribution drift "
                "was detected using this check."
            )


# =========================================================
# FAIRNESS
# =========================================================

elif page == "⚖️ Fairness":

    st.title(
        "⚖️ Fairness Audit"
    )

    st.write(
        """
        This section checks whether the dataset contains
        potentially sensitive attributes that can be used
        for a fairness analysis.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload dataset for fairness audit",
        type=["csv"],
        key="fairness_upload",
    )

    if uploaded_file is None:

        st.info(
            "Upload the dataset used in Experiment 5."
        )

        st.stop()

    df = pd.read_csv(
        uploaded_file
    )

    df = normalize_columns(
        df
    )

    possible_sensitive_features = [
        "gender",
        "sex",
        "race",
        "ethnicity",
        "age_group",
        "religion",
        "region",
    ]

    available = [
        column
        for column in possible_sensitive_features
        if column in df.columns
    ]

    if not available:

        st.warning(
            """
            No conventional sensitive attribute was
            found in the uploaded Layoffs dataset.

            Therefore demographic parity and equalized
            odds cannot be meaningfully calculated from
            this dataset without an appropriate sensitive
            attribute.

            This is a data limitation, not evidence that
            the model is automatically fair.
            """
        )

        st.subheader(
            "Fairness Status"
        )

        st.info(
            "Sensitive attribute unavailable."
        )

    else:

        sensitive = st.selectbox(
            "Sensitive Attribute",
            available
        )

        st.write(
            "Selected attribute:",
            sensitive
        )

        st.dataframe(
            df[sensitive]
            .value_counts()
            .rename(
                "Count"
            ),
            use_container_width=True
        )

        st.info(
            """
            For the final fairness assessment, use the
            Fairlearn metrics generated in Experiment 5.
            """
        )


# =========================================================
# RESPONSIBLE AI
# =========================================================

elif page == "🛡 Responsible AI":

    st.title(
        "🛡 Responsible AI"
    )

    st.subheader(
        "Intended Use"
    )

    st.write(
        """
        The system is intended for educational, analytical
        and research purposes involving historical technology
        layoff data.
        """
    )

    st.subheader(
        "Human Oversight"
    )

    st.write(
        """
        Predictions should not be treated as definitive
        decisions. Human review and additional contextual
        information are required for consequential decisions.
        """
    )

    st.subheader(
        "Fairness"
    )

    st.write(
        """
        Fairness should be evaluated using appropriate
        sensitive attributes and metrics such as demographic
        parity difference and equalized odds difference.
        If sensitive attributes are unavailable, fairness
        conclusions should be treated as limited.
        """
    )

    st.subheader(
        "Privacy"
    )

    st.write(
        """
        The dashboard should not be used to submit
        unnecessary personally identifiable or sensitive
        information.
        """
    )

    st.subheader(
        "Explainability"
    )

    st.write(
        """
        SHAP and LIME provide explanations of model behavior.
        These explanations indicate associations used by the
        model and should not be interpreted as causal evidence.
        """
    )

    st.subheader(
        "Data Drift"
    )

    st.write(
        """
        Changes in the distribution of input data can reduce
        the reliability of predictions. The dashboard provides
        a drift comparison between reference and current data.
        """
    )

    st.subheader(
        "Limitations"
    )

    st.write(
        """
        The model is based on historical data. Economic
        conditions, industries, company characteristics and
        labor-market behavior can change over time.

        Historical data can also contain missing values,
        measurement errors and historical biases.
        """
    )

    st.subheader(
        "Responsible Use Checklist"
    )

    st.checkbox(
        "Use the model for analytical/educational purposes",
        value=True
    )

    st.checkbox(
        "Review predictions using human judgment",
        value=True
    )

    st.checkbox(
        "Avoid entering unnecessary personal information",
        value=True
    )

    st.checkbox(
        "Review model performance and drift regularly",
        value=True
    )

    st.checkbox(
        "Consider fairness and dataset limitations",
        value=True
    )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Tech Layoff Severity Prediction"
)

st.sidebar.caption(
    "Experiments 1–8 | ADS Project"
)
