import streamlit as str
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Konfigurasi Halaman Web
str.set_page_config(page_title="Prediksi Nilai Hunian California", layout="centered")

str.title("🏠 Estimasi Nilai Hunian Menggunakan LightGBM")
str.write("Aplikasi ini memprediksi harga rumah berdasarkan faktor geografis dan sosial ekonomi.")

# 1. Load data & siapkan model secara cepat di background
@str.cache_data
def load_and_train():
    df = pd.read_csv('housing.csv')
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    
    # Ambil list kolom asli sebelum encoding untuk referensi UI
    ocean_options = df['ocean_proximity'].unique().tolist()
    
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True, dtype=int)
    
    X = df.drop(columns=['median_house_value'])
    y = df['median_house_value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = lgb.LGBMRegressor(n_estimators=150, learning_rate=0.05, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X.columns.tolist(), ocean_options

model, scaler, feature_columns, ocean_options = load_and_train()

# 2. Membuat Form Input di UI Streamlit
str.header("Masukkan Karakteristik Wilayah Hunian:")

# Input Faktor Geografis
col1, col2 = str.columns(2)
with col1:
    longitude = str.number_input("Longitude (Bujur)", value=-122.23)
    latitude = str.number_input("Latitude (Lintang)", value=37.88)
with col2:
    ocean_prox = str.selectbox("Kedekatan dengan Pantai/Laut", options=ocean_options)
    housing_age = str.number_input("Median Umur Bangunan (Tahun)", min_value=1, value=41)

# Input Faktor Sosial Ekonomi & Fisik
str.write("---")
col3, col4 = str.columns(2)
with col3:
    med_income = str.number_input("Median Pendapatan (dalam puluhan ribu USD)", value=8.32)
    population = str.number_input("Total Populasi di Area Cluster", min_value=1, value=322)
with col4:
    total_rooms = str.number_input("Total Kamar", min_value=1, value=880)
    total_bedrooms = str.number_input("Total Kamar Tidur", min_value=1, value=129)
    households = str.number_input("Total Kepala Keluarga", min_value=1, value=126)

# 3. Pemrosesan Data Input untuk Prediksi
if str.button("🔮 Estimasi Nilai Jual Hunian", type="primary"):
    # Buat dataframe kosong dengan struktur kolom yang persis sama dengan training data
    input_data = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Isi nilai numerik dasar
    input_data['longitude'] = longitude
    input_data['latitude'] = latitude
    input_data['housing_median_age'] = housing_age
    input_data['total_rooms'] = total_rooms
    input_data['total_bedrooms'] = total_bedrooms
    input_data['population'] = population
    input_data['households'] = households
    input_data['median_income'] = med_income
    
    # Atur One-Hot Encoding untuk ocean_proximity secara dinamis
    dummy_col = f"ocean_proximity_{ocean_prox}"
    if dummy_col in input_data.columns:
        input_data[dummy_col] = 1
        
    # Lakukan scaling pada data input baru
    input_scaled = scaler.transform(input_data)
    
    # Prediksi
    prediction = model.predict(input_scaled)[0]
    
    # Tampilkan Hasil
    str.success(f"### 💵 Estimasi Nilai Hunian: **${prediction:,.2f}**")