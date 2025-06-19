import streamlit as st
import numpy as np
import pandas as pd
import pickle

# === Load Model Components ===
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# === Streamlit UI ===
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🧠 Prediksi Tingkat Obesitas")
st.markdown("Masukkan informasi berikut:")

# Input data
user_input = {
    'Gender': st.selectbox("Gender", ["Female", "Male"]),
    'Age': st.slider("Age", 10, 100, 25),
    'Height': st.number_input("Height (meter)", value=1.70, step=0.01),
    'Weight': st.number_input("Weight (kg)", value=70.0, step=0.1),
    'family_history_with_overweight': st.selectbox("Family History with Overweight", ["yes", "no"]),
    'FAVC': st.selectbox("Frequent high calorie food?", ["yes", "no"]),
    'FCVC': st.slider("Vegetable consumption (1-3)", 1.0, 3.0, 2.0),
    'NCP': st.slider("Number of main meals", 1.0, 4.0, 3.0),
    'CAEC': st.selectbox("Eating between meals", ["no", "Sometimes", "Frequently", "Always"]),
    'SMOKE': st.selectbox("Do you smoke?", ["yes", "no"]),
    'CH2O': st.slider("Water intake (liter)", 1.0, 3.0, 2.0),
    'SCC': st.selectbox("Calories monitor?", ["yes", "no"]),
    'FAF': st.slider("Physical activity (hrs/week)", 0.0, 3.0, 1.0),
    'TUE': st.slider("Time using technology (hrs/day)", 0.0, 2.0, 1.0),
    'CALC': st.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"]),
    'MTRANS': st.selectbox("Transportation", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])
}

# Prediction
if st.button("🔍 Prediksi"):
    input_df = pd.DataFrame([user_input])
    input_encoded = pd.get_dummies(input_df)

    # Tambahkan kolom yang hilang
    for col in features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Pastikan urutan sesuai training
    input_encoded = input_encoded[features]

    # Scaling
    input_scaled = scaler.transform(input_encoded)

    # Predict
    y_pred = model.predict(input_scaled)[0]
    label = label_encoder.inverse_transform([y_pred])[0]

    st.success(f"Hasil Prediksi: **{label}**")

    # Debug
    st.subheader("📌 Kolom Input (Encoded):")
    st.write(input_encoded)
    st.caption(f"Shape: {input_encoded.shape} — Expected: {len(features)} features")
