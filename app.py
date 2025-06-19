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

# === Input Form ===
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

# === Predict Button ===
if st.button("🔍 Prediksi"):
    # Step 1: Convert to DataFrame
    df_input = pd.DataFrame([user_input])

    # Step 2: One-hot encode
    df_encoded = pd.get_dummies(df_input)

    # Step 3: Align with training features
    for col in features:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[features]

    # Step 4: Scaling
    input_scaled = scaler.transform(df_encoded)

    # Step 5: Predict
    pred_class = model.predict(input_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_class])[0]

    # === Output ===
    st.success(f"📊 Hasil Prediksi: **{pred_label}**")

    # Optional: Debug Info
    st.markdown("### 🛠 DEBUG INFO")
    st.write(f"Jumlah fitur input: {len(df_encoded.columns)} (Expected: {len(features)})")
    missing = [col for col in features if col not in df_encoded.columns]
    st.write("Fitur yang tidak ada:", missing)
    st.write("Isi input encoded:")
    st.dataframe(df_encoded)
    st.write("Kelas hasil prediksi:")
    st.write(label_encoder.classes_)
