import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Synthetic Training Data: [Heart Rate, SpO2, Systolic BP, Respiratory Rate]
# SpO2 < 90% or Extreme HR -> Red (2)
# Abnormal Vitals -> Yellow (1)
# Normal Vitals -> Green (0)

X = np.array([
    [72, 98, 120, 16],  # Normal -> Green (0)
    [80, 97, 118, 18],  # Normal -> Green (0)
    [110, 93, 135, 22], # Warning -> Yellow (1)
    [55, 94, 100, 20],  # Warning -> Yellow (1)
    [145, 85, 85, 30],  # Critical -> Red (2)
    [160, 88, 170, 28], # Critical -> Red (2)
    [40, 82, 70, 8],    # Critical -> Red (2)
])

# Labels: 0 = Green (Stable), 1 = Yellow (Urgent), 2 = Red (Critical)
y = np.array([0, 0, 1, 1, 2, 2, 2])

# Train a Random Forest Model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Save the trained model to disk
joblib.dump(model, "triage_model.pkl")
print("✅ Model trained and saved as triage_model.pkl")