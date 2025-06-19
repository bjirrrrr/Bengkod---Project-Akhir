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

# === Ambil input dari user sesuai raw data ===
user_input = {
    'Gender': st.selectbox("Gender", ["Male", "Female"]),
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

# === Prediksi ketika tombol ditekan ===
if st.button("🔍 Prediksi"):
    # Buat DataFrame dari input user
    df_input = pd.DataFrame([user_input])

    # One-hot encode dengan drop_first=True agar match dengan pelatihan
    df_encoded = pd.get_dummies(df_input, drop_first=True)

    # Pastikan semua fitur tersedia
    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Urutkan kolom sesuai model
    df_encoded = df_encoded[features]

    # Scale
    input_scaled = scaler.transform(df_encoded)

    # Prediksi
    y_pred = model.predict(input_scaled)[0]
    y_label = label_encoder.inverse_transform([y_pred])[0]

    st.success(f"Hasil Prediksi: **{y_label}**")

    # Debug info
    st.subheader("Debug Info:")
    st.write("Input Features:", df_encoded)
    st.write("Scaled Input:", input_scaled)
    st.write("Predicted Class:", y_pred)
    st.write("Label (decoded):", y_label)
