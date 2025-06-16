import streamlit as st
import numpy as np
import pickle
import os

model_path = '/content/drive/MyDrive/ObesityApp/model/'

# Load model, scaler, label encoder
model = pickle.load(open(model_path + 'model.pkl', 'rb'))
scaler = pickle.load(open(model_path + 'scaler.pkl', 'rb'))
le = pickle.load(open(model_path + 'label_encoder.pkl', 'rb'))

st.title("🍔 Prediksi Tingkat Obesitas")

def user_input():
    age = st.slider("Usia", 10, 100, 25)
    height = st.number_input("Tinggi (meter)", value=1.70)
    weight = st.number_input("Berat (kg)", value=65.0)
    fcvc = st.slider("Konsumsi Sayur (1-3)", 1, 3, 2)
    ncp = st.slider("Jumlah Makan (1-4)", 1, 4, 3)
    ch2o = st.slider("Air per Hari (1-3)", 1, 3, 2)
    faf = st.slider("Aktivitas Fisik (0-3)", 0, 3, 1)
    tue = st.slider("Waktu Layar (0-2)", 0, 2, 1)

    favc = st.selectbox("Makan Tinggi Kalori?", ["yes", "no"])
    smoke = st.selectbox("Merokok?", ["yes", "no"])
    scc = st.selectbox("Konsumsi Sayur Konsisten?", ["yes", "no"])
    fh = st.selectbox("Riwayat Keluarga Gemuk?", ["yes", "no"])
    caec = st.selectbox("Kebiasaan Camilan?", ["no", "Sometimes", "Frequently", "Always"])
    calc = st.selectbox("Minum Alkohol?", ["no", "Sometimes", "Frequently", "Always"])
    mtrans = st.selectbox("Transportasi", ["Walking", "Bike", "Motorbike", "Automobile", "Public_Transportation"])
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])

    manual = np.array([[age, height, weight, fcvc, ncp, ch2o, faf, tue,
                        favc == 'yes', smoke == 'yes', scc == 'yes', fh == 'yes',
                        caec == 'Always', caec == 'Frequently', caec == 'Sometimes',
                        calc == 'Always', calc == 'Frequently', calc == 'Sometimes',
                        mtrans == 'Automobile', mtrans == 'Bike', mtrans == 'Motorbike',
                        mtrans == 'Public_Transportation', gender == 'Male']])
    return manual

input_data = user_input()
scaled = scaler.transform(input_data)

if st.button("Prediksi"):
    pred = model.predict(scaled)
    label = le.inverse_transform(pred)
    st.success(f"Hasil prediksi: **{label[0]}**")
