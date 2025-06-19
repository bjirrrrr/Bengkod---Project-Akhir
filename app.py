import streamlit as st
import numpy as np
import pandas as pd
import pickle

# === Load Komponen Model ===
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
st.title("🧠 Prediksi Tingkat Obesitas")
st.markdown("Masukkan informasi berikut:")

# Input
user_input = {
    'Gender': st.selectbox("Gender", ["Male", "Female"]),
    'Age': st.slider("Age", 10, 100, 25),
    'Height': st.number_input("Height (m)", value=1.70, step=0.01),
    'Weight': st.number_input("Weight (kg)", value=70.0, step=0.1),
    'family_history_with_overweight': st.selectbox("Family History Overweight", ["yes", "no"]),
    'FAVC': st.selectbox("Frequent high calorie food?", ["yes", "no"]),
    'FCVC': st.slider("Vegetable Consumption (1-3)", 1.0, 3.0, 2.0),
    'NCP': st.slider("Number of Meals", 1.0, 4.0, 3.0),
    'CAEC': st.selectbox("Eating between meals", ["no", "Sometimes", "Frequently", "Always"]),
    'SMOKE': st.selectbox("Do you smoke?", ["yes", "no"]),
    'CH2O': st.slider("Water Intake (L)", 1.0, 3.0, 2.0),
    'SCC': st.selectbox("Calories Monitor?", ["yes", "no"]),
    'FAF': st.slider("Physical Activity (hrs/week)", 0.0, 3.0, 1.0),
    'TUE': st.slider("Time Using Technology (hrs/day)", 0.0, 2.0, 1.0),
    'CALC': st.selectbox("Alcohol Consumption", ["no", "Sometimes", "Frequently", "Always"]),
    'MTRANS': st.selectbox("Transport", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])
}

# === Proses Prediksi ===
if st.button("🔍 Prediksi"):
    df_input = pd.DataFrame([user_input])
    df_encoded = pd.get_dummies(df_input)

    # Tambahkan semua fitur yang mungkin hilang
    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Pastikan urutan kolom tepat
    df_encoded = df_encoded[features]

    # Scaling
    input_scaled = scaler.transform(df_encoded)

    # Prediksi
    pred_class = model.predict(input_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    # Output
    st.success(f"Hasil Prediksi: **{pred_label}**")

    # DEBUG
    st.markdown("### 🔍 DEBUG INFO")
    st.markdown(f"Jumlah fitur input: {df_encoded.shape[1]} (Expected: {len(features)})")
    missing = [f for f in features if f not in df_encoded.columns]
    st.markdown(f"Fitur yang tidak ada: {missing}")
    st.dataframe(df_encoded)

