# 🚑 Connected Ambulance System with AI-Assisted Triage

An AI-powered emergency response and telemetry dashboard built for the **Emerging Technologies Hackathon 2026** (Problem Statement: **PS14**).

## 🌟 Key Features
- **Real-Time Patient Telemetry:** Live tracking of Heart Rate, $SpO_2$, Blood Pressure, and Respiratory Rate.
- **Predictive AI Triage:** Classifies patient urgency into Green (Stable), Yellow (Urgent), or Red (Critical).
- **Emergency Room Alerts:** Triggers instant alerts so doctors can prepare trauma bays before the ambulance arrives.
- **Live Dispatch Tracking:** Interactive map showing ambulance dispatch position and route.

## 🛠️ Tech Stack
- **Language:** Python
- **UI / Dashboard:** Streamlit
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Maps:** Folium

## 🚀 How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt

## 🧠 How the AI Model is Trained

The predictive triage engine is built using a complete Machine Learning workflow contained in `train_model.py`. 

1. **Synthetic Data Generation:** We generate 1,000 realistic patient telemetry records spanning normal, urgent, and critical vital ranges (Heart Rate, SpO2, Systolic BP, Respiratory Rate).
2. **Clinical Rules Engine:** The data is labeled using standard medical early warning scoring (like MEWS) to classify patients into three target categories:
   - `0`: Green / Stable
   - `1`: Yellow / Urgent
   - `2`: Red / Critical
3. **Model Training:** A **Random Forest Classifier** (`RandomForestClassifier` from `scikit-learn`) is trained on 80% of the data and validated on the remaining 20% to ensure high accuracy.
4. **Model Export:** The fully trained model is exported as a serialized `triage_model.pkl` file, which the Streamlit dashboard loads instantly to run real-time predictions.
