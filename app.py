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

# === UI Streamlit ===
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🌟 Prediksi Tingkat Obesitas")
st.markdown("Masukkan informasi berikut:")

# === Form Input User ===
user_input = {
    'Gender': st.selectbox("Jenis Kelamin", ["Male", "Female"]),
    'Age': st.slider("Umur", 10, 100, 25),
    'Height': st.number_input("Tinggi Badan (meter)", value=1.70, step=0.01),
    'Weight': st.number_input("Berat Badan (kg)", value=70.0, step=0.1),
    'family_history_with_overweight': st.selectbox("Riwayat keluarga obesitas?", ["yes", "no"]),
    'FAVC': st.selectbox("Sering makan berkalori tinggi?", ["yes", "no"]),
    'FCVC': st.slider("Konsumsi sayur (1-3)", 1.0, 3.0, 2.0),
    'NCP': st.slider("Jumlah makan/hari (1-4)", 1.0, 4.0, 3.0),
    'CAEC': st.selectbox("Ngemil di antara waktu makan?", ["no", "Sometimes", "Frequently", "Always"]),
    'SMOKE': st.selectbox("Merokok?", ["yes", "no"]),
    'CH2O': st.slider("Konsumsi air putih (liter)", 1.0, 3.0, 2.0),
    'SCC': st.selectbox("Mengontrol asupan kalori?", ["yes", "no"]),
    'FAF': st.slider("Aktivitas fisik (jam/minggu)", 0.0, 3.0, 1.0),
    'TUE': st.slider("Durasi pakai teknologi (jam/hari)", 0.0, 2.0, 1.0),
    'CALC': st.selectbox("Konsumsi alkohol?", ["no", "Sometimes", "Frequently", "Always"]),
    'MTRANS': st.selectbox("Transportasi utama", ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])
}

if st.button("🔍 Prediksi"):
    # Buat DataFrame dari input
    df_input = pd.DataFrame([user_input])
    df_encoded = pd.get_dummies(df_input)

    # Pastikan semua kolom dari training ada
    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    df_encoded = df_encoded[features]

    # Debugging - tampilkan input sebelum dan sesudah scaling
    st.subheader("🧾 Data setelah one-hot encoding:")
    st.dataframe(df_encoded)

    # Scaling
    scaled_input = scaler.transform(df_encoded)

    # Debugging - nilai input setelah scaling
    st.subheader("📏 Data setelah scaling:")
    st.write(scaled_input)

    # Prediksi
    pred_class = model.predict(scaled_input)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    # Probabilitas prediksi
    probas = model.predict_proba(scaled_input)[0]
    classes = label_encoder.inverse_transform(np.arange(len(probas)))
    prob_dict = dict(zip(classes, probas.round(3)))

    st.subheader("🔢 Probabilitas Prediksi:")
    st.write(prob_dict)

    # Hasil akhir
    st.success(f"Hasil Prediksi: **{pred_label}**")
