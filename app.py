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

# === UI ===
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🌟 Prediksi Tingkat Obesitas")
st.markdown("Masukkan informasi berikut untuk memprediksi level obesitas:")

# Input User
user_input = {
    'Gender': st.selectbox("Gender", ["Male", "Female"]),
    'Age': st.slider("Umur", 10, 100, 25),
    'Height': st.number_input("Tinggi Badan (meter)", value=1.70, step=0.01),
    'Weight': st.number_input("Berat Badan (kg)", value=70.0, step=0.1),
    'family_history_with_overweight': st.selectbox("Riwayat keluarga kegemukan?", ["yes", "no"]),
    'FAVC': st.selectbox("Sering konsumsi makanan berkalori tinggi?", ["yes", "no"]),
    'FCVC': st.slider("Frekuensi konsumsi sayur (1-3)", 1.0, 3.0, 2.0),
    'NCP': st.slider("Jumlah makan per hari (1-4)", 1.0, 4.0, 3.0),
    'CAEC': st.selectbox("Makan di luar waktu makan?", ["no", "Sometimes", "Frequently", "Always"]),
    'SMOKE': st.selectbox("Apakah merokok?", ["yes", "no"]),
    'CH2O': st.slider("Konsumsi air (liter)", 1.0, 3.0, 2.0),
    'SCC': st.selectbox("Mengontrol kalori?", ["yes", "no"]),
    'FAF': st.slider("Aktivitas fisik (jam/minggu)", 0.0, 3.0, 1.0),
    'TUE': st.slider("Waktu di depan layar (jam/hari)", 0.0, 2.0, 1.0),
    'CALC': st.selectbox("Konsumsi alkohol", ["no", "Sometimes", "Frequently", "Always"]),
    'MTRANS': st.selectbox("Transportasi utama", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]),
}

if st.button("🔍 Prediksi"):
    # Buat DataFrame
    input_df = pd.DataFrame([user_input])

    # One-hot encode
    input_encoded = pd.get_dummies(input_df)

    # Tambahkan kolom yang hilang agar sesuai dengan fitur saat training
    for col in features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[features]  # Reorder columns

    # Skala
    input_scaled = scaler.transform(input_encoded)

    # Prediksi
    y_pred = model.predict(input_scaled)[0]
    label = label_encoder.inverse_transform([y_pred])[0]

    st.success(f"🎯 Hasil Prediksi: **{label}**")
