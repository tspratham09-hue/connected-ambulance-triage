import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Generate Synthetic Patient Telemetry (1,000 samples)
np.random.seed(42)
n_samples = 1000

heart_rate = np.random.randint(40, 180, size=n_samples)
spo2 = np.random.randint(70, 101, size=n_samples)
systolic_bp = np.random.randint(70, 200, size=n_samples)
respiratory_rate = np.random.randint(8, 40, size=n_samples)

X = pd.DataFrame(
    {
        "Heart Rate": heart_rate,
        "SpO2": spo2,
        "Systolic BP": systolic_bp,
        "Respiratory Rate": respiratory_rate,
    }
)


# 2. Define Clinical Triage Rules
def assign_triage(row):
    # RED (Critical): Severe hypoxia, low BP, or extreme heart/respiratory rates
    if (
        row["SpO2"] < 90
        or row["Heart Rate"] > 130
        or row["Heart Rate"] < 45
        or row["Systolic BP"] < 85
        or row["Respiratory Rate"] > 30
    ):
        return 2  # Critical / Red
    # YELLOW (Urgent): Borderline abnormal vitals
    elif (
        row["SpO2"] < 95
        or row["Heart Rate"] > 100
        or row["Heart Rate"] < 55
        or row["Systolic BP"] > 140
        or row["Respiratory Rate"] > 22
    ):
        return 1  # Urgent / Yellow
    # GREEN (Stable): Normal ranges
    else:
        return 0  # Stable / Green


y = X.apply(assign_triage, axis=1)

# 3. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model Training Complete!")
print(f"📊 Model Accuracy: {accuracy * 100:.2f}%\n")
print("Detailed Classification Metrics:")
print(classification_report(y_test, y_pred, target_names=["Green", "Yellow", "Red"]))

# 6. Save Trained Model File
joblib.dump(model, "triage_model.pkl")
print("💾 Model saved to 'triage_model.pkl'")