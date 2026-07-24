import joblib
import random
import time
import streamlit as st
import folium
import pandas as pd
import plotly.express as px

from streamlit_folium import st_folium

# Page configuration
st.set_page_config(page_title="AI Connected Ambulance System", layout="wide")

st.title("🚑 Connected Ambulance System & AI Triage Dashboard")
st.markdown("Real-time telemetry and predictive triage for hospital emergency preparedness.")

# --- NEW: Initialize Session State for the Chart ---
if 'vitals_history' not in st.session_state:
    st.session_state['vitals_history'] = pd.DataFrame(columns=['Time', 'Heart Rate', 'SpO2'])
    st.session_state['time_counter'] = 0

# Load Trained Model
@st.cache_resource
def load_model():
    return joblib.load("triage_model.pkl")

model = load_model()

# Sidebar: Controls to Simulate Ambulance Vitals
st.sidebar.header("🕹️ Ambulance Control Panel")

# 1. Initialize default values securely
if 'hr' not in st.session_state:
    st.session_state['hr'] = 75
    st.session_state['spo2'] = 98
    st.session_state['bp'] = 120
    st.session_state['rr'] = 16

# 2. Button updates the session state directly
if st.sidebar.button("🚨 Simulate Critical Incident"):
    st.session_state['hr'] = 145
    st.session_state['spo2'] = 86
    st.session_state['bp'] = 85
    st.session_state['rr'] = 30

# 3. Sliders use the 'key' argument to instantly sync with the button
hr = st.sidebar.slider("Heart Rate (BPM)", 40, 180, key='hr')
spo2 = st.sidebar.slider("Oxygen Saturation (SpO2 %)", 70, 100, key='spo2')
bp = st.sidebar.slider("Systolic BP (mmHg)", 60, 200, key='bp')
rr = st.sidebar.slider("Respiratory Rate", 8, 40, key='rr')

# --- NEW: Update the History Dataframe ---
new_data = pd.DataFrame({
    'Time': [st.session_state['time_counter']],
    'Heart Rate': [hr],
    'SpO2': [spo2]
})
# Append new reading and increase time counter
st.session_state['vitals_history'] = pd.concat([st.session_state['vitals_history'], new_data], ignore_index=True)
st.session_state['time_counter'] += 1

# Keep only the last 20 readings so the chart doesn't get cluttered
if len(st.session_state['vitals_history']) > 20:
    st.session_state['vitals_history'] = st.session_state['vitals_history'].iloc[-20:]

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
        # Automatically play warning alarm when status is RED
        st.audio("alarm.mp3", autoplay=True)
    elif prediction == 1:
        st.warning(f"### NOTICE: {status_text}")
    else:
        st.success(f"### STATUS: {status_text}")
        
    st.info(f"**Recommended Action:** {hospital_action}")

st.markdown("---")

# --- NEW: Display the Live Chart ---
st.subheader("📈 Live Patient Vitals Trend")
fig = px.line(st.session_state['vitals_history'], x='Time', y=['Heart Rate', 'SpO2'], 
              title="Continuous Telemetry Feed (Last 20 Readings)", markers=True)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- Map View Upgrade ---
st.subheader("📍 Live Ambulance Dispatch & Routing")

# Simulated Coordinates (Bangalore example)
ambulance_lat, ambulance_lon = 12.9716, 77.5946
hospital_lat, hospital_lon = 12.9850, 77.5900

# Center map between ambulance and hospital
m = folium.Map(location=[12.9780, 77.5920], zoom_start=14)

# 1. Hospital Marker & Emergency Radius
folium.Marker(
    [hospital_lat, hospital_lon], 
    popup="Central City Hospital (Trauma Center)", 
    icon=folium.Icon(color="green", icon="h-square", prefix="fa")
).add_to(m)

folium.Circle(
    location=[hospital_lat, hospital_lon],
    radius=1200, # 1.2 km catchment radius
    color="green",
    fill=True,
    fill_opacity=0.1,
    tooltip="Emergency Preparedness Zone"
).add_to(m)

# 2. Ambulance Marker (Turns Red in Critical Status)
folium.Marker(
    [ambulance_lat, ambulance_lon], 
    popup="Ambulance #04 (Active Dispatch)", 
    icon=folium.Icon(color="red" if prediction == 2 else "blue", icon="ambulance", prefix="fa")
).add_to(m)

# 3. Draw the active route (Dashed line)
folium.PolyLine(
    locations=[[ambulance_lat, ambulance_lon], [hospital_lat, hospital_lon]],
    color="red" if prediction == 2 else "blue",
    weight=4,
    dash_array='10, 10',
    tooltip="ETA: 4 Minutes"
).add_to(m)

# Display on Streamlit
st_folium(m, width=1100, height=450)