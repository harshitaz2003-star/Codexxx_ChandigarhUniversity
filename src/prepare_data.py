import pandas as pd

# Load datasets
ehr = pd.read_csv("data/ehr.csv")
wearable = pd.read_csv("data/wearable.csv")

# Sort wearable data correctly
wearable = wearable.sort_values(
    ["patient_id", "time_index"]
)

# Add glucose trend features
# Each reading represents 15 minutes.
wearable["glucose_change_15min"] = (
    wearable.groupby("patient_id")["glucose"]
    .diff(1)
)

wearable["glucose_change_30min"] = (
    wearable.groupby("patient_id")["glucose"]
    .diff(2)
)

# Replace missing values at the beginning of each patient's
# time series with zero
wearable["glucose_change_15min"] = (
    wearable["glucose_change_15min"].fillna(0)
)

wearable["glucose_change_30min"] = (
    wearable["glucose_change_30min"].fillna(0)
)

# Merge wearable data with static EHR information
fused = wearable.merge(
    ehr,
    on="patient_id",
    how="left"
)

# 2 hours = 8 readings × 15 minutes
fused["future_glucose_2h"] = (
    fused.groupby("patient_id")["glucose"]
    .shift(-8)
)

# Define the prediction target
fused["glucose_spike_2h"] = (
    fused["future_glucose_2h"] >= 180
).astype(int)

# Remove rows where future glucose is unavailable
fused = fused.dropna(
    subset=["future_glucose_2h"]
)

# Save the new fused dataset
fused.to_csv(
    "data/fused_dataset.csv",
    index=False
)

print("Fused dataset created successfully!")
print("Total records:", len(fused))

print("\nNew trend features:")
print("glucose_change_15min")
print("glucose_change_30min")

print("\nTarget distribution:")
print(fused["glucose_spike_2h"].value_counts())

print("\nFirst 5 records:")
print(fused.head())