import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Load fused dataset
df = pd.read_csv("data/fused_dataset.csv")

# Features from BOTH EHR and wearable data
features = [
    # Static EHR features
    "age",
    "bmi",
    "hba1c",
    "fasting_glucose",
    "systolic_bp",
    "diabetes_history",
    "hypertension",
    "family_history_diabetes",
    "genetic_risk",

    # Dynamic wearable features
    "time_index",
    "glucose",
    "heart_rate",
    "hrv",
    "steps",
    "sleep_hours",

    # New glucose trend features
    "glucose_change_15min",
    "glucose_change_30min"
]

X = df[features]
y = df["glucose_spike_2h"]

# Split by patient
# This prevents the same patient appearing in both
# training and testing datasets.
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=df["patient_id"]
    )
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

# Get probability of glucose spike
probabilities = model.predict_proba(X_test)[:, 1]

# Lower threshold to improve detection of spike events
threshold = 0.30

predictions = (
    probabilities >= threshold
).astype(int)

# Evaluation
accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probabilities
)

print("\n==============================")
print("DIGITAL TWIN MODEL RESULTS")
print("==============================")

print(f"Accuracy :  {accuracy * 100:.2f}%")
print(f"Precision:  {precision * 100:.2f}%")
print(f"Recall   :  {recall * 100:.2f}%")
print(f"F1 Score :  {f1 * 100:.2f}%")
print(f"ROC-AUC  :  {auc:.3f}")

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions
))

# Feature importance
importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance.to_string(index=False))

# Save trained model
joblib.dump(
    model,
    "models/glucose_spike_model.pkl"
)

print("\nModel saved successfully!")
print("models/glucose_spike_model.pkl")