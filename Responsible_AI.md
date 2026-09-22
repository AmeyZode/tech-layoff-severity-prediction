# Responsible AI Report

## 1. Project Overview

The Tech Layoff Severity Prediction system is a machine
learning application designed to classify the severity of
employee layoffs.

The model predicts whether the percentage of employees
affected by layoffs is below 50% or 50% and above.

---

## 2. Intended Use

The system is intended for:

- Educational experimentation
- Data analysis
- Machine learning research
- Demonstration of predictive modeling
- Exploratory analysis of historical layoff data

The system should not be used as the sole basis for
employment, financial, legal, or organizational decisions.

---

## 3. Fairness

Fairness was evaluated using the Fairlearn framework
where applicable.

The analysis considers whether model performance varies
between groups represented by available sensitive
attributes.

Fairness metrics should be interpreted together with
overall model performance and dataset limitations.

---

## 4. Privacy

The project uses historical dataset information.

Personal identifying information should not be entered
into the prediction interface.

The dashboard should only collect information required
for the prediction task.

---

## 5. Consent

The system is intended to operate on appropriately
obtained datasets.

Data should be collected, processed, and used according
to the permissions and terms applicable to the original
dataset.

---

## 6. Transparency

The application provides:

- Prediction probabilities
- Model performance metrics
- SHAP-based explanations
- Data drift checks
- Documentation of model limitations

These mechanisms help users understand how the model
behaves.

---

## 7. Explainability

SHAP and LIME were used to investigate model predictions.

SHAP provides global and local explanations of model
behavior, while LIME provides explanations for individual
predictions.

Explanations describe model behavior and should not be
interpreted as causal relationships.

---

## 8. Limitations

The model is trained on historical layoff data.

Historical patterns may not represent future labor-market
conditions.

Potential limitations include:

- Missing values
- Data quality issues
- Historical bias
- Changes in economic conditions
- Distribution changes
- Limited representation of some groups
- Dependence on available features

---

## 9. Human Oversight

Predictions should be treated as analytical outputs rather
than definitive decisions.

Important decisions should involve human review and
additional contextual information.

---

## 10. Security

The API and dashboard should be deployed with appropriate
security controls.

Sensitive information should not be submitted through
the application.

Dependencies should be maintained and updated when
compatible with the trained model.

---

## 11. Model Monitoring

The system can be monitored using:

- Prediction performance
- Data drift
- Feature distributions
- Model metrics
- Fairness metrics

Significant changes should trigger additional evaluation.

---

## 12. Responsible Use

The model should be used primarily for educational,
research, and analytical purposes.

Users should consider model uncertainty, dataset
limitations, fairness considerations, and changing
real-world conditions before interpreting predictions.
