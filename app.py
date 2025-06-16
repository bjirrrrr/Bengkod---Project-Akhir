import streamlit as st
import numpy as np
import pickle

# === Load model & tools ===
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

with open("features.pkl", "rb") as f:
    full_feature_names = pickle.load(f)

# === Streamlit UI ===
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("🧠 Prediksi Tingkat Obesitas")

st.markdown("Masukkan informasi berikut:")

# === Input User ===
age = st.slider("Usia", 10, 100, 25)
height = st.number_input("Tinggi badan (meter)", value=1.70, format="%.2f")
weight = st.number_input("Berat badan (kg)", value=65.0, format="%.1f")
fcvc = st.slider("Frekuensi makan sayur (0-3)", 0.0, 3.0, 2.0)
ncp = st.slider("Jumlah makan utama/hari", 1.0, 4.0, 3.0)
ch2o = st.slider("Konsumsi air/hari (liter)", 0.0, 3.0, 2.0)
faf = st.slider("Aktivitas fisik (jam/minggu)", 0.0, 3.0, 1.0)
tue = st.slider("Waktu layar (jam)", 0.0, 2.0, 1.0)

gender = st.selectbox("Jenis Kelamin", ["Female", "Male"])
family_history = st.selectbox("Riwayat keluarga obesitas", ["no", "yes"])
smoke = st.selectbox("Merokok?", ["no", "yes"])
scc = st.selectbox("Ikut perawatan diet (SCC)?", ["no", "yes"])
caec = st.selectbox("Konsumsi makanan tinggi kalori", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Jenis transportasi", ["Public_Transportation", "Walking", "Bike", "Automobile", "Motorbike"])

# === Saat tombol ditekan ===
if st.button("🔍 Prediksi"):
    # Buat fitur input dengan one-hot encoding manual
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

    # Pastikan input sesuai dengan urutan fitur saat training
    input_array = np.array([[input_dict.get(col, 0) for col in full_feature_names]])
    input_scaled = scaler.transform(input_array)
    pred = model.predict(input_scaled)
    pred_label = le.inverse_transform(pred)[0]

    st.success(f"🎯 Prediksi Tingkat Obesitas: **{pred_label}**")
