import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load EHR data
df = pd.read_csv("data/ehr.csv")

# Convert medication to numbers
encoder = LabelEncoder()
df["medication_encoded"] = encoder.fit_transform(df["medication"])

# Features
features = [
    "age",
    "bmi",
    "hba1c",
    "fasting_glucose",
    "systolic_bp",
    "diabetes_history",
    "hypertension",
    "family_history_diabetes",
    "genetic_risk"
]

X = df[features]

# Create diabetes risk target
y = (
    (df["hba1c"] >= 6.5) |
    (df["fasting_glucose"] >= 126) |
    (df["diabetes_history"] == 1)
).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model trained successfully!")
print("Accuracy:", round(accuracy * 100, 2), "%")

# Save model
joblib.dump(model, "models/diabetes_model.pkl")

print("Model saved to models/diabetes_model.pkl")