import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --------------------------
# PAGE CONFIG
# --------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🍔",
    layout="wide"
)

# --------------------------
# CUSTOM CSS
# --------------------------
st.markdown("""
<style>
.main { padding: 1rem; }
.pred-box {
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
    font-size: 25px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------
# LOAD MODEL
# --------------------------
@st.cache_resource
def load_model():
    # Looks for model in same folder as this script
    base = os.path.dirname(os.path.abspath(__file__))
    return joblib.load(os.path.join(base, "linearmodel.pkl"))

model = load_model()

# --------------------------
# HEADER
# --------------------------
st.title("🍔 Food Delivery Time Predictor")
st.markdown("Predict delivery time using distance, traffic, weather and courier details.")

# --------------------------
# INPUT SECTION
# --------------------------
col1, col2 = st.columns(2)

with col1:
    distance = st.slider("Distance (km)",           min_value=1.0,  max_value=30.0, value=10.0)
    prep_time = st.slider("Preparation Time (min)", min_value=1,    max_value=60,   value=15)
    experience = st.slider("Courier Experience (years)", min_value=0.0, max_value=10.0, value=2.0)

with col2:
    weather  = st.selectbox("Weather",       ["Clear", "Foggy", "Rainy", "Windy"])
    traffic  = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
    time_day = st.selectbox("Time of Day",   ["Morning", "Afternoon", "Evening", "Night"])
    vehicle  = st.selectbox("Vehicle Type",  ["Bike", "Scooter"])

# --------------------------
# PREDICT BUTTON
# --------------------------
if st.button("🚀 Predict Delivery Time", use_container_width=True):

    # ── Step 1: build raw input as a DataFrame (same column names as training) ──
    input_df = pd.DataFrame([{
        "Distance_km":              distance,
        "Preparation_Time_min":     prep_time,
        "Courier_Experience_yrs":   experience,
        "Weather":                  weather,
        "Traffic_Level":            traffic,
        "Time_of_Day":              time_day,
        "Vehicle_Type":             vehicle,
    }])

    # ── Step 2: one-hot encode exactly as training did ─────────────────────────
    input_encoded = pd.get_dummies(input_df)

    # ── Step 3: align to the exact columns the model was trained on ────────────
    #    - adds any missing dummy columns as 0
    #    - drops any extra columns not seen in training
    #    - preserves column order
    input_encoded = input_encoded.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # ── Step 4: predict ────────────────────────────────────────────────────────
    prediction = model.predict(input_encoded)[0]

    st.markdown(
        f"""
        <div class="pred-box">
        Estimated Delivery Time<br>
        {prediction:.1f} Minutes
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------
# FOOTER
# --------------------------
st.markdown("---")
st.caption("Built with Streamlit ❤️")