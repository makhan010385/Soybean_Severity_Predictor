import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Model_CSV1.csv")
    df["Mean_RH"] = (df["Max_Humidity"] + df["Min_Humidity"]) / 2
    return df

df = load_data()

# Title
st.title("Soybean Disease Severity Predictor")
st.markdown("""
This app predicts the severity of **Anthracnose, Rhizoctonia Aerial Blight (RAB), Charcoal Rot, and Yellow Mosaic Virus (YMV)** in soybean
based on weather conditions and selected variety, using regression models derived from field research.
""")

# Sidebar inputs
st.sidebar.header("Input Parameters")

variety_list = [col for col in df.columns if col.startswith("JS") or col in ["Shivalik", "Punjab1", "PK -472", "Bragg", "Monetta", "NRC-7", "PK-262", "Gaurav"]]
selected_variety = st.sidebar.selectbox("Select Soybean Variety", variety_list)

mean_rh = st.sidebar.slider("Mean Relative Humidity (%)", 50.0, 100.0, 85.0)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 200.0, 50.0)
min_temp = st.sidebar.slider("Minimum Temperature (°C)", 15.0, 30.0, 24.0)

# Prediction formulas for different diseases

def predict_anthracnose(mean_rh, rainfall, min_temp):
    return max(0, round(-45.0 + 0.4 * mean_rh - 0.05 * rainfall + 1.2 * min_temp, 2))

def predict_rab(mean_rh, rainfall, min_temp):
    return max(0, round(-58.8 + 0.433 * mean_rh - 0.0498 * rainfall + 1.354 * min_temp, 2))

def predict_charcoal_rot(mean_rh, rainfall, min_temp):
    return max(0, round(-30.0 + 0.35 * mean_rh - 0.04 * rainfall + 1.0 * min_temp, 2))

def predict_ymv(mean_rh, rainfall, min_temp):
    return max(0, round(-50.0 + 0.38 * mean_rh - 0.03 * rainfall + 1.1 * min_temp, 2))

# Predict severity scores
severity_anthracnose = predict_anthracnose(mean_rh, rainfall, min_temp)
severity_rab = predict_rab(mean_rh, rainfall, min_temp)
severity_charcoal = predict_charcoal_rot(mean_rh, rainfall, min_temp)
severity_ymv = predict_ymv(mean_rh, rainfall, min_temp)

# Display results
st.subheader("Predicted Disease Severity (PDI)")
st.metric(label="Anthracnose", value=f"{severity_anthracnose} %")
st.metric(label="Rhizoctonia Aerial Blight (RAB)", value=f"{severity_rab} %")
st.metric(label="Charcoal Rot", value=f"{severity_charcoal} %")
st.metric(label="Yellow Mosaic Virus (YMV)", value=f"{severity_ymv} %")

# Display filtered real data for selected variety
st.subheader(f"Historical Records for {selected_variety}")
filtered = df[(df[selected_variety].notna()) & (df[selected_variety] > 0)]
st.dataframe(filtered[["Year", "SMW", selected_variety, "Max_Temp", "Min_Temp", "Rainfall", "Max_Humidity", "Min_Humidity"]].reset_index(drop=True))

# Plot historical PDI for selected variety
st.subheader(f"{selected_variety} Weekly PDI Over Time")
fig, ax = plt.subplots()
filtered.groupby("SMW")[selected_variety].mean().plot(ax=ax, marker='o')
ax.set_xlabel("Standard Meteorological Week (SMW)")
ax.set_ylabel("Average PDI")
ax.set_title(f"{selected_variety} Disease Progression")
st.pyplot(fig)

# Note
st.markdown("**Note:** This model uses simplified regression equations for different soybean diseases and is intended for indicative forecasting only.")
