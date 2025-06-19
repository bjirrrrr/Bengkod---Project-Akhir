import streamlit as st
import pickle
import pandas as pd

# Fungsi untuk memuat komponen model (mem-cache agar hanya sekali load)
@st.cache(allow_output_mutation=True)
def load_components():
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    label_enc = pickle.load(open('label_encoder.pkl', 'rb'))
    features = pickle.load(open('features.pkl', 'rb'))
    return model, scaler, label_enc, features

model, scaler, label_enc, features = load_components()

st.title("Prediksi Kategori Obesitas")

# Input numerik
age = st.number_input("Usia", min_value=0, max_value=120, value=30)
height = st.number_input("Tinggi (meter)", min_value=0.0, max_value=3.0, value=1.7, format="%.2f")
weight = st.number_input("Berat (kg)", min_value=0.0, max_value=500.0, value=70.0, format="%.1f")
fcvc = st.number_input("Frekuensi makan sayuran per hari (FCVC)", min_value=0, max_value=5, value=2)
ncp = st.number_input("Jumlah makanan per hari (NCP)", min_value=1, max_value=5, value=3)
ch2o = st.number_input("Konsumsi air per hari (liter, CH2O)", min_value=0.0, max_value=10.0, value=2.0, format="%.1f")
faf = st.number_input("Frekuensi aktivitas fisik per minggu (FAF)", min_value=0, max_value=7, value=1)
tue = st.number_input("Waktu penggunaan alat elektronik per hari (jam, TUE)", min_value=0.0, max_value=24.0, value=2.0, format="%.1f")

# Input kategori
gender = st.selectbox("Gender", ["Female", "Male"])
calc = st.selectbox("Frekuensi konsumsi fast food (CALC)", 
                    ["Always", "Frequently", "Sometimes", "no"])
favc = st.selectbox("Sering makan camilan (FAVC)?", ["yes", "no"])
scc = st.selectbox("Riwayat merokok (SCC)?", ["yes", "no"])
smoke = st.selectbox("Merokok sekarang (SMOKE)?", ["yes", "no"])
fhwo = st.selectbox("Keluarga obesitas (yes/no)?", ["yes", "no"])
caec = st.selectbox("Konsumsi alkohol (CAEC)", 
                    ["Always", "Frequently", "Sometimes", "no"])
mtrans = st.selectbox("Transportasi ke tempat kerja (MTRANS)", 
                      ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])

# Susun data input menjadi DataFrame
input_data = {
    'Age': age,
    'Height': height,
    'Weight': weight,
    'FCVC': fcvc,
    'NCP': ncp,
    'CH2O': ch2o,
    'FAF': faf,
    'TUE': tue,
    'Gender': gender,
    'CALC': calc,
    'FAVC': favc,
    'SCC': scc,
    'SMOKE': smoke,
    'family_history_with_overweight': fhwo,
    'CAEC': caec,
    'MTRANS': mtrans
}
input_df = pd.DataFrame([input_data])

# One-hot encoding dan penyelarasan kolom
df_encoded = pd.get_dummies(input_df)
df_aligned = df_encoded.reindex(columns=features, fill_value=0)

# Standardisasi
X_scaled = scaler.transform(df_aligned)

# Prediksi dan konversi label
prediction_num = model.predict(X_scaled)
prediction_label = label_enc.inverse_transform(prediction_num)[0]

# Tampilkan hasil prediksi dan fitur input yang digunakan
st.write(f"**Prediksi kategori obesitas:** {prediction_label}")
st.write("**Fitur input (nama kolom) yang dikirim ke model:**")
st.write(df_aligned.columns.tolist())
