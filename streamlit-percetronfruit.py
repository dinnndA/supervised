import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load model Perceptron
model_file = 'perceptronfruit.pkl'
with open(model_file, 'rb') as f:
    model = pickle.load(f)

# Load scaler (jika diperlukan)
scaler_file = 'scaler_perceptron.pkl'  # Scaler harus disimpan saat training
with open(scaler_file, 'rb') as f:
    scaler = pickle.load(f)

# Fungsi untuk prediksi
def predict_fruit(features):
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    return prediction[0]

# Konfigurasi Streamlit
st.title("Aplikasi Prediksi Buah")
st.write("Masukkan fitur buah untuk memprediksi jenis buah.")

# Input pengguna
input_features = []
columns = ['Fitur1', 'Fitur2', 'Fitur3', 'Fitur4']  # Sesuaikan dengan kolom fitur
for col in columns:
    value = st.number_input(f"Masukkan nilai untuk {col}:", value=0.0)
    input_features.append(value)

# Prediksi
if st.button("Prediksi"):
    result = predict_fruit(input_features)
    st.success(f"Model memprediksi jenis buah: {result}")
