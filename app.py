import streamlit as st
import numpy as np
import pickle

# === Load model, scaler, and label encoder ===
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# === Streamlit UI ===
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🌟 Prediksi Tingkat Obesitas")

st.markdown("Masukkan data berikut untuk memprediksi tingkat obesitas:")

# === Form input ===
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi (meter)", value=1.70, format="%.2f")
weight = st.number_input("Berat (kg)", value=65.0, format="%.1f")
fcvc = st.slider("Frekuensi konsumsi sayur (0-3)", 0.0, 3.0, 2.0)
ncp = st.slider("Jumlah makanan utama per hari (1-4)", 1.0, 4.0, 3.0)
ch2o = st.slider("Konsumsi air (liter/hari)", 0.0, 3.0, 2.0)
faf = st.slider("Aktivitas fisik mingguan (jam)", 0.0, 3.0, 1.0)
tue = st.slider("Waktu layar (jam)", 0.0, 2.0, 1.0)

# === Kategori (one-hot otomatis di model) ===
gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
family_history = st.selectbox("Riwayat keluarga obesitas", ["yes", "no"])
smoke = st.selectbox("Apakah merokok?", ["yes", "no"])
scc = st.selectbox("Mengikuti perawatan diet?", ["yes", "no"])
caec = st.selectbox("Konsumsi makanan berkalori tinggi?", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportasi", ["Public_Transportation", "Walking", "Bike", "Automobile", "Motorbike"])

# === Prediksi tombol ===
if st.button("🔍 Prediksi"):
    # Buat DataFrame dummy dari input pengguna
    input_dict = {
        "Age": age, "Height": height, "Weight": weight, "FCVC": fcvc,
        "NCP": ncp, "CH2O": ch2o, "FAF": faf, "TUE": tue,
        "Gender_Male": 1 if gender == "Male" else 0,
        "family_history_with_overweight_yes": 1 if family_history == "yes" else 0,
        "SMOKE_yes": 1 if smoke == "yes" else 0,
        "SCC_yes": 1 if scc == "yes" else 0,
        "CAEC_Sometimes": 1 if caec == "Sometimes" else 0,
        "CAEC_Frequently": 1 if caec == "Frequently" else 0,
        "CAEC_Always": 1 if caec == "Always" else 0,
        "MTRANS_Walking": 1 if mtrans == "Walking" else 0,
        "MTRANS_Bike": 1 if mtrans == "Bike" else 0,
        "MTRANS_Automobile": 1 if mtrans == "Automobile" else 0,
        "MTRANS_Motorbike": 1 if mtrans == "Motorbike" else 0,
    }

    # Pastikan semua kolom ada (urutan sama seperti X_train)
    dummy = np.zeros((1, model.n_features_in_))
    for i, col in enumerate(model.feature_names_in_):
        dummy[0, i] = input_dict.get(col, 0)

    # Scaling dan prediksi
    dummy_scaled = scaler.transform(dummy)
    pred = model.predict(dummy_scaled)
    pred_label = le.inverse_transform(pred)[0]

    st.success(f"🧠 Prediksi Tingkat Obesitas: **{pred_label}**")
