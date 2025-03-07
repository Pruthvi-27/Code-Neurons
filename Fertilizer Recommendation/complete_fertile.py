# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 17:50:28 2025

@author: nitee
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load dataset
file_path = "C:\\Users\\nitee\\Desktop\\Fertilizer Prediction.csv"
df = pd.read_csv(file_path)

# Encode categorical variables
le_soil = LabelEncoder()
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])

le_crop = LabelEncoder()
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])

# Define features and target
X = df[['Temparature', 'Moisture', 'Soil Type', 'Crop Type', 'Nitrogen', 'Phosphorous', 'Potassium']]

y = df['Fertilizer Name']

# Train or load Random Forest model
model_path = "rf_model.pkl"
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

# Streamlit UI
#st.set_page_config(page_title="Fertilizer Recommendation", layout="centered")
st.title("🌱 Fertilizer Recommendation System")
st.write("Select soil and crop details to get the best fertilizer recommendation.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    temperature = st.number_input("Temperature (°C)", min_value=10, max_value=50, value=25, step=1)

    moisture = st.number_input(" Moisture (%)", min_value=10, max_value=100, value=50, step=1)

with col2:
    soil_type = st.selectbox("Soil Type", le_soil.classes_)
    crop_type = st.selectbox(" Crop Type", le_crop.classes_)

n = st.slider(" Nitrogen (N)", min_value=0, max_value=100, value=50)
p = st.slider(" Phosphorous (P)", min_value=0, max_value=100, value=50)
k = st.slider(" Potassium (K)", min_value=0, max_value=100, value=50)

# Output container
output_container = st.container()

def recommend_fertilizer():
    input_data = np.array([
        [temperature, moisture, le_soil.transform([soil_type])[0], le_crop.transform([crop_type])[0], n, p, k]
    ])
    fertilizer = model.predict(input_data)[0]
    with output_container:
        st.success(f" Recommended Fertilizer: {fertilizer}")

def check_soil_fertility():
    fertility_score = np.mean([n, p, k, moisture])
    with output_container:
        if fertility_score > 60:
            st.success(" Soil is highly fertile!")
        elif fertility_score > 30:
            st.warning("⚠️ Soil is moderately fertile.")
        else:
            st.error("❌ Soil is not fertile. Consider improving soil quality.")

st.button(" Check Fertilizer", on_click=recommend_fertilizer)
st.button(" Check Soil Fertility", on_click=check_soil_fertility)