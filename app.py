import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Title
st.title("🌾 AgroCast Pro - Crop Yield Prediction")

# Input fields
N = st.number_input("Nitrogen (N)")
P = st.number_input("Phosphorus (P)")
K = st.number_input("Potassium (K)")
temperature = st.number_input("Temperature")
humidity = st.number_input("Humidity")
ph = st.number_input("pH")
rainfall = st.number_input("Rainfall")

# Feature Engineering
Soil_Index = (N + P + K) / 3
Climate_Index = rainfall * humidity

# Load dataset & train model
df = pd.read_csv("Crop_recommendation.csv")

df["Yield"] = (
    0.3 * df["N"] +
    0.2 * df["rainfall"] -
    0.1 * df["temperature"] +
    np.random.normal(0, 10, len(df))
)

df = df.drop("label", axis=1)

df["Soil_Index"] = (df["N"] + df["P"] + df["K"]) / 3
df["Climate_Index"] = df["rainfall"] * df["humidity"]

X = df.drop("Yield", axis=1)
y = df["Yield"]

model = RandomForestRegressor()
model.fit(X, y)

# Prediction button
if st.button("Predict Yield"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall, Soil_Index, Climate_Index]])
    prediction = model.predict(input_data)
    
    st.success(f"🌾 Predicted Yield: {prediction[0]:.2f}")