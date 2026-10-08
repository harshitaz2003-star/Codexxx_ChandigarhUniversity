# 🏥 AI-Enabled Digital Twin for Early Prediction of Glucose Spikes

Team Codexxx | Chandigarh University

An AI-enabled healthcare Digital Twin prototype that combines simulated Electronic Health Record (EHR) information with simulated wearable health data to estimate the probability of a glucose spike within the next 2 hours.

## Problem Statement

Glucose levels can change over time due to meals, physical activity, and individual health conditions. Monitoring current glucose levels alone may not provide enough warning of a future spike.

This project combines patient health profiles, laboratory information, and dynamic wearable measurements to build a machine-learning prototype for early glucose-spike risk prediction.

## Proposed Solution

The system includes four main stages:

1. Data Generation:create synthetic EHR and wearable datasets.
2. Data Fusion: Combine patient profiles with time-series health measurements.
3. Machine Learning:Use a Random Forest classifier to estimate the risk of a glucose spike within 2 hours.
4. Dashboard:Display patient information, glucose trends, model predictions, and feature importance using Streamlit and Plotly.

Important: This is an educational prototype using synthetic data, not a clinically validated medical system.

## Model Performance

The final Random Forest model was evaluated on a held-out synthetic test set.

| Metric | Result |
|---|---:|
| Accuracy | 92.80% |
| Precision | 22.33% |
| Recall | 17.86% |
| F1 Score | 19.85% |
| ROC-AUC | 0.876 |

### Confusion Matrix

```text
[[16175   546]
 [  722   157]]

 ## Technology Stack

### Programming Language
- Python

### Data Processing
- Pandas
- NumPy

### Machine Learning
- Scikit-learn
- Random Forest Classifier

### Visualization
- Plotly
- Matplotlib

### Dashboard
- Streamlit

### Model Saving
- Joblib

## Project Structure

```text
Healthcare_Digital_Twin/
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── ehr.csv
│   ├── wearable.csv
│   ├── fused_dataset.csv
│   ├── generate_ehr.py
│   └── generate_wearable.py
│
├── models/
│   ├── glucose_spike_model.pkl
│   └── train_fused_model.py
│
├── src/
│   └── prepare_data.py
│
├── notebook/
│
├── requirements.txt
│
└── README.md

## How to Run

### 1. Install Dependencies

Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
## Dataset Description

This project uses fully synthetic healthcare data generated locally for prototype development.

### Static EHR Data

The synthetic EHR dataset contains:

- Patient demographics
- Age and sex
- BMI
- HbA1c
- Fasting glucose
- Systolic blood pressure
- Diabetes history
- Hypertension history
- Family history of diabetes
- Genetic risk score
- Medication information

### Dynamic Wearable / CGM Data

The simulated time-series dataset contains:

- Glucose measurements
- Heart rate
- Heart rate variability (HRV)
- Steps
- Sleep duration
- Time index

### Target Variable

The machine-learning target is:

**`glucose_spike_2h`**

A value of `1` indicates that the simulated glucose level is expected to reach **180 mg/dL or higher within the next 2 hours**.

All data is synthetic and contains no real patient information.

## System Architecture

The Digital Twin follows this pipeline:

```text
┌──────────────────────┐
│   Static EHR Data    │
│ Demographics, Labs,  │
│ History, Risk Factors│
└──────────┬───────────┘
           │
           │
           ▼
┌──────────────────────┐
│   Dynamic Data       │
│ CGM, HR, HRV, Steps, │
│ Sleep Measurements   │
└──────────┬───────────┘
           │
           ▼
┌────────────────────────────┐
│       Data Fusion          │
│ EHR + Wearable Time-Series │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│    Feature Engineering     │
│ Glucose Trends + Health    │
│ and Activity Features      │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│    Random Forest Model     │
│ 2-Hour Glucose Spike Risk  │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│      AI Prediction         │
│ Probability of Glucose     │
│ Spike within 2 Hours       │
└────────────┬───────────────┘
             │
             ▼
┌────────────────────────────┐
│    Doctor Dashboard        │
│ Streamlit + Plotly         │
└────────────────────────────┘

Prediction Workflow
Patient EHR information provides the static health profile.
Wearable and CGM data provide dynamic health measurements.
Both datasets are joined using the patient ID.
Glucose trend features are calculated from the time-series data.
The Random Forest model processes the combined features.
The model estimates the probability of a glucose spike within 2 hours.
The Streamlit dashboard presents the patient profile, trends, prediction, and model information.
Architecture Diagram

A detailed visual architecture diagram is provided separately in the project submission materials.

## License

This project is released under the **MIT License**.

The project is intended for educational, research, and prototype demonstration purposes.