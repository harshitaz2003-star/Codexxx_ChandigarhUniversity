import pandas as pd
import numpy as np

np.random.seed(42)

n_patients = 1000
readings_per_patient = 96

data = []

for i in range(1, n_patients + 1):

    patient_id = f"P{i:04d}"

    # Patient-specific baseline
    base_glucose = np.random.normal(110, 12)
    base_heart_rate = np.random.normal(75, 7)
    base_hrv = np.random.normal(50, 8)

    # Some patients have higher probability of glucose spikes
    high_risk = np.random.random() < 0.30

    # Create several predictable spike events during the day
    spike_starts = []

    for possible_start in [24, 32, 52, 60, 72]:
        probability = 0.35 if high_risk else 0.15

        if np.random.random() < probability:
            spike_starts.append(possible_start)

    for j in range(readings_per_patient):

        hour = j / 4

        # Normal meal effect
        meal_effect = 0

        if 8 <= hour < 10:
            meal_effect = 20

        elif 13 <= hour < 15:
            meal_effect = 25

        elif 19 <= hour < 21:
            meal_effect = 20

        # Predictable glucose spike
        spike_effect = 0

        for start in spike_starts:

            distance = j - start

            if 0 <= distance <= 12:

                # Spike rises and then gradually falls
                if distance <= 4:
                    spike_effect = max(
                        spike_effect,
                        distance * 15
                    )
                else:
                    spike_effect = max(
                        spike_effect,
                        max(0, 90 - (distance - 4) * 8)
                    )

        # Glucose
        glucose_value = (
            base_glucose
            + meal_effect
            + spike_effect
            + np.random.normal(0, 4)
        )

        glucose_value = max(60, glucose_value)

        # Activity
        if 7 <= hour < 9 or 17 <= hour < 19:
            steps = np.random.randint(150, 700)
        else:
            steps = np.random.randint(0, 180)

        # Heart rate increases with activity
        heart_rate_value = (
            base_heart_rate
            + steps / 120
            + np.random.normal(0, 2)
        )

        # HRV
        hrv_value = (
            base_hrv
            - (heart_rate_value - 70) * 0.25
            + np.random.normal(0, 2)
        )

        # Sleep
        sleep_hours = np.random.normal(6.5, 0.4)

        data.append([
            patient_id,
            j,
            round(glucose_value, 1),
            round(heart_rate_value, 1),
            round(hrv_value, 1),
            steps,
            round(sleep_hours, 1)
        ])

wearable = pd.DataFrame(
    data,
    columns=[
        "patient_id",
        "time_index",
        "glucose",
        "heart_rate",
        "hrv",
        "steps",
        "sleep_hours"
    ]
)

wearable.to_csv("data/wearable.csv", index=False)

print("Synthetic wearable dataset created successfully!")
print("Total records:", len(wearable))

print("\nGlucose statistics:")
print(wearable["glucose"].describe())

print("\nNumber of readings >= 180 mg/dL:")
print((wearable["glucose"] >= 180).sum())
