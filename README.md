# 🚑 Connected Ambulance System & AI Triage Dashboard

An end-to-end intelligent emergency medical response system that connects ambulances directly to emergency rooms in real-time. By leveraging AI-driven vital predictive triage, standardizing telemetry data streams, and mapping live dispatch routes, emergency medical teams can prepare resuscitation bays prior to patient arrival.

---

## 🌟 Key Features

* **⚡ Real-Time Patient Telemetry:** Continuous streaming of vital parameters including Heart Rate (BPM), Oxygen Saturation ($SpO_2$), Systolic Blood Pressure, and Respiratory Rate.
* **🤖 Predictive AI Triage:** Machine Learning classification model categorizes incoming patients into **🟢 STABLE**, **🟡 URGENT**, or **🔴 CRITICAL** status instantly.
* **🚨 Automated Trauma Bay Alerts:** Trigger visual alerts and automated audio alarms when a patient enters critical status.
* **📈 Interactive Vitals Trend Chart:** Live Plotly visualization tracking real-time fluctuations across patient vitals.
* **📍 Live Dispatch & Route Tracking:** Dynamic Folium spatial map displaying ambulance positioning, target hospital trauma centers, and active emergency zones.

---

## 🛠️ Tech Stack

* **Frontend / Dashboard:** Streamlit, Plotly, Streamlit-Folium
* **Machine Learning:** Scikit-Learn, Joblib, Pandas
* **Mapping & GIS:** Folium
* **Language:** Python 3.10+

---

## 💻 Local Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/tspratham09-hue/ambulance_triage.git](https://github.com/tspratham09-hue/ambulance_triage.git)
   cd ambulance_triage