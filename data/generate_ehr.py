import pandas as pd
import numpy as np

np.random.seed(42)

n_patients = 1000

ehr = pd.DataFrame({
    "patient_id": [f"P{i:04d}" for i in range(1, n_patients + 1)],
    "age": np.random.randint(18, 75, n_patients),
    "sex": np.random.choice(["Male", "Female"], n_patients),
    "bmi": np.round(np.random.normal(27, 4, n_patients), 1),
    "hba1c": np.round(np.random.normal(5.8, 0.8, n_patients), 1),
    "fasting_glucose": np.round(np.random.normal(105, 20, n_patients), 0),
    "systolic_bp": np.random.randint(100, 160, n_patients),
    "diabetes_history": np.random.choice([0, 1], n_patients, p=[0.7, 0.3]),
    "hypertension": np.random.choice([0, 1], n_patients, p=[0.65, 0.35]),
    "family_history_diabetes": np.random.choice([0, 1], n_patients, p=[0.6, 0.4]),
    "genetic_risk": np.round(np.random.uniform(0, 1, n_patients), 2),
    "medication": np.random.choice(
        ["None", "Metformin", "Insulin", "Other"],
        n_patients,
        p=[0.65, 0.2, 0.05, 0.1]
    )
})

ehr["bmi"] = ehr["bmi"].clip(16, 45)
ehr["hba1c"] = ehr["hba1c"].clip(4.0, 10.0)
ehr["fasting_glucose"] = ehr["fasting_glucose"].clip(60, 250)

ehr.to_csv("data/ehr.csv", index=False)

print("Synthetic EHR dataset created successfully!")
print(f"Number of patients: {len(ehr)}")
print("\nFirst 5 patients:")
print(ehr.head())