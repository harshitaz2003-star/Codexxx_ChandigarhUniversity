import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Healthcare Digital Twin",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# LOAD DATA AND MODEL
# ============================================================

df = pd.read_csv("data/fused_dataset.csv")

model = joblib.load(
    "models/glucose_spike_model.pkl"
)

# ============================================================
# TITLE
# ============================================================

st.title("🏥 Healthcare Digital Twin")

st.subheader(
    "AI-Enabled Early Prediction of Glucose Spikes"
)

st.write(
    """
    This prototype combines static Electronic Health Record (EHR)
    information with dynamic wearable/CGM data to estimate the
    probability of a glucose spike occurring within the next 2 hours.
    """
)

st.divider()

# ============================================================
# PATIENT SELECTION
# ============================================================

st.sidebar.header("Patient Selection")

patient_ids = sorted(
    df["patient_id"].unique()
)

patient_id = st.sidebar.selectbox(
    "Select Patient",
    patient_ids
)

# Get selected patient's records
patient_data = df[
    df["patient_id"] == patient_id
].copy()

# Select current time point
max_time = int(
    patient_data["time_index"].max()
)

time_index = st.sidebar.slider(
    "Select Time Point",
    min_value=0,
    max_value=max_time,
    value=20
)

current = patient_data[
    patient_data["time_index"] == time_index
]

if current.empty:
    st.warning(
        "No data available for this time point."
    )
    st.stop()

current = current.iloc[0]

# ============================================================
# PATIENT PROFILE
# ============================================================

st.header("👤 Patient Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Patient ID",
        patient_id
    )

with col2:
    st.metric(
        "Age",
        int(current["age"])
    )

with col3:
    st.metric(
        "BMI",
        f"{current['bmi']:.1f}"
    )

with col4:
    st.metric(
        "HbA1c",
        f"{current['hba1c']:.1f}%"
    )

# ============================================================
# STATIC EHR DATA
# ============================================================

st.header("📋 Static EHR Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write(
        "**Sex:**",
        current["sex"]
    )

with col2:
    st.write(
        "**Fasting Glucose:**",
        f"{current['fasting_glucose']:.0f} mg/dL"
    )

with col3:
    st.write(
        "**Systolic BP:**",
        f"{current['systolic_bp']:.0f} mmHg"
    )

with col4:
    st.write(
        "**Diabetes History:**",
        "Yes"
        if current["diabetes_history"] == 1
        else "No"
    )

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write(
        "**Hypertension:**",
        "Yes"
        if current["hypertension"] == 1
        else "No"
    )

with col2:
    st.write(
        "**Family History:**",
        "Yes"
        if current["family_history_diabetes"] == 1
        else "No"
    )

with col3:
    st.write(
        "**Genetic Risk:**",
        f"{current['genetic_risk']:.2f}"
    )

with col4:
    st.write(
        "**Medication:**",
        current["medication"]
        if pd.notna(current["medication"])
        else "None"
    )

# ============================================================
# DYNAMIC WEARABLE DATA
# ============================================================

st.header("⌚ Dynamic Health Data")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Glucose",
        f"{current['glucose']:.1f} mg/dL"
    )

with col2:
    st.metric(
        "Heart Rate",
        f"{current['heart_rate']:.1f} bpm"
    )

with col3:
    st.metric(
        "HRV",
        f"{current['hrv']:.1f} ms"
    )

with col4:
    st.metric(
        "Steps",
        int(current["steps"])
    )

with col5:
    st.metric(
        "Sleep",
        f"{current['sleep_hours']:.1f} h"
    )

# ============================================================
# GLUCOSE TREND
# ============================================================

st.header("📈 Glucose Trend")

recent_data = patient_data[
    patient_data["time_index"] <= time_index
].tail(32)

fig = px.line(
    recent_data,
    x="time_index",
    y="glucose",
    markers=True,
    labels={
        "time_index": "Time Index",
        "glucose": "Glucose (mg/dL)"
    },
    title="Recent Glucose Measurements"
)

fig.add_hline(
    y=180,
    line_dash="dash",
    annotation_text="Spike Threshold: 180 mg/dL"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ============================================================
# GLUCOSE TREND FEATURES
# ============================================================

st.header("📊 Glucose Trend Indicators")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "15-Minute Glucose Change",
        f"{current['glucose_change_15min']:+.1f} mg/dL"
    )

with col2:
    st.metric(
        "30-Minute Glucose Change",
        f"{current['glucose_change_30min']:+.1f} mg/dL"
    )

# ============================================================
# AI PREDICTION
# ============================================================

features = [
    "age",
    "bmi",
    "hba1c",
    "fasting_glucose",
    "systolic_bp",
    "diabetes_history",
    "hypertension",
    "family_history_diabetes",
    "genetic_risk",
    "time_index",
    "glucose",
    "heart_rate",
    "hrv",
    "steps",
    "sleep_hours",
    "glucose_change_15min",
    "glucose_change_30min"
]

input_data = pd.DataFrame(
    [[current[f] for f in features]],
    columns=features
)

probability = model.predict_proba(
    input_data
)[0][1]

threshold = 0.30

prediction = int(
    probability >= threshold
)

# ============================================================
# AI RISK RESULT
# ============================================================

st.header("🤖 AI Prediction")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "2-Hour Glucose Spike Risk",
        f"{probability * 100:.1f}%"
    )

with col2:

    if prediction == 1:
        st.error(
            "⚠️ Elevated Predicted Spike Risk"
        )
    else:
        st.success(
            "✅ Lower Predicted Spike Risk"
        )

st.progress(
    min(probability, 1.0)
)

st.caption(
    "Prediction threshold used by the prototype: 30%"
)

# ============================================================
# DIGITAL TWIN SUMMARY
# ============================================================

st.header("🧬 Digital Twin Summary")

st.write(
    f"""
    **Patient:** {patient_id}

    The digital twin combines the patient's static EHR information
    with dynamic physiological and activity data.

    At the selected time point:

    - Current glucose: **{current['glucose']:.1f} mg/dL**
    - 15-minute glucose change: **{current['glucose_change_15min']:+.1f} mg/dL**
    - 30-minute glucose change: **{current['glucose_change_30min']:+.1f} mg/dL**
    - Heart rate: **{current['heart_rate']:.1f} bpm**
    - HRV: **{current['hrv']:.1f}**
    - Steps: **{int(current['steps'])}**

    The machine-learning model uses these dynamic measurements together
    with the patient's EHR profile to estimate the probability of a
    glucose spike within the next 2 hours.
    """
)
# ============================================================
# AI EXPLAINABILITY
# ============================================================

st.header("🔎 AI Prediction Factors")

st.write(
    """
    The following features had the greatest influence on the
    Random Forest model during training. They represent the
    information the model uses to distinguish higher-risk and
    lower-risk glucose-spike patterns.
    """
)

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
).head(8)

fig_importance = px.bar(
    feature_importance,
    x="Importance",
    y="Feature",
    orientation="h",
    title="Top Model Features",
    labels={
        "Importance": "Relative Importance",
        "Feature": "Feature"
    }
)

fig_importance.update_layout(
    yaxis={"categoryorder": "total ascending"}
)

st.plotly_chart(
    fig_importance,
    use_container_width=True
)

st.caption(
    "Feature importance shows the model's learned global feature "
    "importance. It does not represent a clinical causal relationship."
)
# ============================================================
# MODEL INFORMATION
# ============================================================

st.header("🔬 Model Information")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "ROC-AUC",
        "0.876"
    )

with col3:
    st.metric(
        "F1 Score",
        "19.85%"
    )

with col4:
    st.metric(
        "Prediction Window",
        "2 Hours"
    )

# ============================================================
# DISCLAIMER
# ============================================================

st.divider()

st.info(
    """
    ⚠️ Prototype and educational demonstration only.

    This system uses synthetic data and is not a medical diagnostic
    system. The reported model performance is based on synthetic
    prototype data and should not be interpreted as clinical validation.
    """
)

