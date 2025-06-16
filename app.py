import streamlit as st
import numpy as np
import pandas as pd
import pickle

# === Load all saved model components ===
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("features.pkl", "rb") as f:
    features = pickle.load(f)

# === Streamlit UI ===
st.title("Prediksi Tingkat Obesitas 🚶‍♂️")

st.markdown("Masukkan data berikut untuk prediksi:")

# Input sesuai dengan fitur asli sebelum one-hot
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 100, 25)
height = st.number_input("Height (meter)", value=1.70, step=0.01)
weight = st.number_input("Weight (kg)", value=70.0, step=0.5)
family_history = st.selectbox("Family History with Overweight", ["yes", "no"])
favc = st.selectbox("Do you eat high caloric food frequently? (FAVC)", ["yes", "no"])
fcvc = st.slider("Frequency of Vegetable Consumption (FCVC)", 1.0, 3.0, 2.0)
ncp = st.slider("Number of Meals (NCP)", 1.0, 4.0, 3.0)
caec = st.selectbox("Consumption of food between meals (CAEC)", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Do you smoke?", ["yes", "no"])
ch2o = st.slider("Water consumption (liters)", 1.0, 3.0, 2.0)
scc = st.selectbox("Do you monitor your calorie intake? (SCC)", ["yes", "no"])
faf = st.slider("Physical Activity (hrs/week)", 0.0, 3.0, 1.0)
tue = st.slider("Time using technology (hrs/day)", 0.0, 2.0, 1.0)
calc = st.selectbox("Frequency of alcohol consumption (CALC)", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportation Used", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

# === Predict button ===
if st.button("🔍 Prediksi"):
    # Buat dict input user
    input_dict = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history,
        'FAVC': favc,
        'FCVC': fcvc,
        'NCP': ncp,
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': ch2o,
        'SCC': scc,
        'FAF': faf,
        'TUE': tue,
        'CALC': calc,
        'MTRANS': mtrans
    }

    # Ubah jadi DataFrame dan encode one-hot
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)

    # Tambah kolom yang hilang agar match dengan training
    for col in features:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[features]  # urutkan

    # Scaling
    input_scaled = scaler.transform(input_encoded)

    # Prediksi
    pred_class = model.predict(input_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    st.success(f"Hasil Prediksi: **{pred_label}**")
