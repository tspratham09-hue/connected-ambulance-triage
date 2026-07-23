import joblib
import random
import time
import streamlit as st
import folium
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(page_title="AI Connected Ambulance System", layout="wide")

st.title("🚑 Connected Ambulance System & AI Triage Dashboard")
st.markdown("Real-time telemetry and predictive triage for hospital emergency preparedness.")

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load("triage_model.pkl")

model = load_model()

# Sidebar: Controls to Simulate Ambulance Vitals
st.sidebar.header("🕹️ Ambulance Control Panel")

# Quick emergency simulation button
if st.sidebar.button("🚨 Simulate Critical Incident"):
    st.session_state['hr'] = 145
    st.session_state['spo2'] = 86
    st.session_state['bp'] = 85
    st.session_state['rr'] = 30
else:
    st.session_state.setdefault('hr', 75)
    st.session_state.setdefault('spo2', 98)
    st.session_state.setdefault('bp', 120)
    st.session_state.setdefault('rr', 16)

hr = st.sidebar.slider("Heart Rate (BPM)", 40, 180, st.session_state['hr'])
spo2 = st.sidebar.slider("Oxygen Saturation (SpO2 %)", 70, 100, st.session_state['spo2'])
bp = st.sidebar.slider("Systolic BP (mmHg)", 60, 200, st.session_state['bp'])
rr = st.sidebar.slider("Respiratory Rate", 8, 40, st.session_state['rr'])

# AI Prediction
input_data = [[hr, spo2, bp, rr]]
prediction = model.predict(input_data)[0]

# Mapping Triage Levels
triage_status = {
    0: ("🟢 STABLE (GREEN)", "st.success", "Standard ER Admission"),
    1: ("🟡 URGENT (YELLOW)", "st.warning", "Prepare Observation Bed"),
    2: ("🔴 CRITICAL (RED)", "st.error", "IMMEDIATE RESUSCITATION REQUIRED — Prepare Trauma Bay!")
}

status_text, alert_style, hospital_action = triage_status[prediction]

# Main Dashboard Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Incoming Patient Telemetry")
    st.metric(label="Heart Rate", value=f"{hr} BPM")
    st.metric(label="SpO2 Level", value=f"{spo2} %", delta="- Danger" if spo2 < 90 else "Normal")
    st.metric(label="Blood Pressure", value=f"{bp} mmHg")
    st.metric(label="Respiratory Rate", value=f"{rr} / min")

with col2:
    st.subheader("🎯 AI Triage Classification")
    
    if prediction == 2:
        st.error(f"### ALERT: {status_text}")
    elif prediction == 1:
        st.warning(f"### NOTICE: {status_text}")
    else:
        st.success(f"### STATUS: {status_text}")
        
    st.info(f"**Recommended Action:** {hospital_action}")

st.markdown("---")

# Map View
st.subheader("📍 Live Ambulance Dispatch Tracking")
# Coordinates for a sample route (e.g., city center to hospital)
ambulance_lat, ambulance_lon = 12.9716, 77.5946

m = folium.Map(location=[ambulance_lat, ambulance_lon], zoom_start=13)
folium.Marker(
    [ambulance_lat, ambulance_lon], 
    popup="Ambulance #04 (ETA: 7 mins)", 
    icon=folium.Icon(color="red" if prediction == 2 else "blue", icon="plus")
).add_to(m)

st_folium(m, width=1100, height=350)