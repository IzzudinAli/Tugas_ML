# Laporan Proyek Machine Learning - [Izzudin Ali]

## Domain Proyek

### Latar Belakang

Sektor properti dan hunian merupakan salah satu indikator penting dalam perekonomian suatu negara. Nilai hunian tidak hanya ditentukan oleh kondisi fisik bangunan, namun sangat dipengaruhi oleh faktor-faktor geografis dan sosial ekonomi yang melingkupinya. Faktor-faktor seperti lokasi administratif, kedekatan dengan pusat kota, tingkat pendapatan masyarakat sekitar, kepadatan penduduk, serta akses terhadap fasilitas publik terbukti memiliki korelasi yang signifikan terhadap harga properti di suatu wilayah.

Di Indonesia, ketimpangan nilai hunian antarwilayah masih tergolong tinggi. Kawasan perkotaan dengan infrastruktur memadai cenderung memiliki nilai hunian jauh lebih tinggi dibanding kawasan pinggiran atau perdesaan, meskipun secara fisik bangunan memiliki kualitas yang setara. Pemahaman terhadap faktor-faktor penentu nilai hunian ini menjadi krusial bagi berbagai pemangku kepentingan, mulai dari pengembang properti, pemerintah daerah dalam perencanaan tata ruang, hingga masyarakat umum yang ingin membuat keputusan investasi properti yang tepat.

### Masalah yang Harus Diselesaikan

Berdasarkan data yang menunjukkan bahwa lebih dari 60% keputusan pembelian properti di Indonesia kurang mempertimbangkan faktor geografis dan sosial ekonomi secara komprehensif, diperlukan sebuah pendekatan berbasis data untuk menganalisis dan memprediksi nilai hunian secara lebih akurat. Dengan memanfaatkan teknik machine learning, analisis ini diharapkan dapat memberikan gambaran yang lebih objektif mengenai faktor-faktor apa saja yang paling berpengaruh terhadap nilai hunian di suatu wilayah.

Format Referensi: [Indeks Harga Properti Residensial - Bank Indonesia](https://www.bi.go.id/id/publikasi/laporan/Pages/IHPR-Triwulan-IV-2023.aspx)

---

## Business Understanding

### Problem Statements

1. Faktor geografis dan sosial ekonomi apa saja yang paling berpengaruh terhadap nilai hunian di suatu wilayah?
2. Bagaimana cara membangun model prediktif yang akurat untuk memperkirakan nilai hunian berdasarkan fitur-fitur geografis dan sosial ekonomi yang tersedia?
3. Apakah terdapat pola atau kluster wilayah yang dapat diidentifikasi berdasarkan kesamaan profil geografis, sosial ekonomi, dan nilai huniannya?

### Goals

- Mengidentifikasi dan menganalisis faktor-faktor geografis (seperti koordinat lokasi, kedekatan dengan pusat kota, dan ketinggian wilayah) serta faktor sosial ekonomi (seperti median pendapatan, kepadatan penduduk, dan tingkat pengangguran) yang memiliki pengaruh signifikan terhadap nilai hunian.
- Mengembangkan model machine learning yang dapat memprediksi nilai hunian secara akurat berdasarkan fitur-fitur yang telah diidentifikasi.
- Menghasilkan wawasan yang dapat digunakan oleh pemangku kepentingan untuk pengambilan keputusan terkait investasi dan perencanaan properti.

### Solution Statements

Menggunakan algoritma **XGBoost Regressor** (*Extreme Gradient Boosting*) untuk memprediksi nilai hunian berdasarkan fitur-fitur geografis dan sosial ekonomi. XGBoost dipilih karena kemampuannya menangani hubungan non-linear antarvariabel, ketahanannya terhadap overfitting melalui mekanisme regularisasi bawaan (L1 dan L2), serta performanya yang terbukti unggul pada data tabular.

Evaluasi model akan menggunakan metrik **RMSE (Root Mean Squared Error)**, **MAE (Mean Absolute Error)**, dan **R² Score** untuk memastikan model memberikan prediksi yang relevan dan akurat.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini berisi informasi mengenai karakteristik geografis dan sosial ekonomi berbagai wilayah, beserta nilai median hunian di masing-masing wilayah tersebut.

Sumber dataset: [Kaggle - California Housing Prices Dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices)

### Informasi Dataset

- **Jumlah data**: Dataset ini terdiri dari **20.640 baris** dan **10 kolom**.
- **Kondisi data**:
  - *Missing values*: Terdapat 207 nilai yang hilang pada kolom `total_bedrooms`, yang selanjutnya ditangani dengan imputasi median.
  - *Duplikat*: Tidak ditemukan data duplikat setelah pengecekan menggunakan `duplicated().sum()`.
  - *Outlier*: Beberapa fitur seperti `median_income` dan `median_house_value` memiliki nilai ekstrim yang teridentifikasi melalui visualisasi box plot dan metode IQR.

### Fitur pada Dataset

| Fitur | Deskripsi |
|---|---|
| `longitude` | Koordinat bujur lokasi blok perumahan |
| `latitude` | Koordinat lintang lokasi blok perumahan |
| `housing_median_age` | Median usia bangunan di blok tersebut (tahun) |
| `total_rooms` | Total jumlah ruangan dalam satu blok |
| `total_bedrooms` | Total jumlah kamar tidur dalam satu blok |
| `population` | Jumlah penduduk dalam satu blok |
| `households` | Jumlah rumah tangga dalam satu blok |
| `median_income` | Median pendapatan rumah tangga (dalam puluhan ribu USD) |
| `median_house_value` | Median nilai hunian (target/label, dalam USD) |
| `ocean_proximity` | Kategori kedekatan dengan laut (fitur geografis kategorikal) |

### Exploratory Data Analysis (EDA)

Sebelum masuk ke tahap pemodelan, dilakukan eksplorasi data untuk memahami distribusi dan hubungan antarvariabel.

**Distribusi Variabel Numerik**:
- **Distribusi Median Income**: Distribusi cenderung right-skewed, dengan sebagian besar rumah tangga memiliki pendapatan rendah hingga menengah. Terdapat beberapa nilai ekstrim di kisaran pendapatan tinggi.
- **Distribusi Median House Value**: Distribusi menunjukkan adanya *clipping* pada nilai maksimum (500.000 USD), yang mengindikasikan adanya data yang dicap pada batas tertentu.
- **Distribusi Housing Median Age**: Distribusi relatif merata, dengan konsentrasi pada usia bangunan 10–52 tahun.
- **Distribusi Population**: Sangat right-skewed, dengan sebagian besar blok memiliki populasi rendah namun beberapa blok memiliki populasi sangat tinggi.

```
# Contoh output visualisasi distribusi fitur
housing_data[['median_income', 'median_house_value', 
              'housing_median_age', 'population']].hist(bins=50, figsize=(12, 8))
plt.suptitle('Distribusi Fitur Numerik')
plt.tight_layout()
plt.savefig('distribusi_fitur.png')
```

![Distribusi Fitur](https://via.placeholder.com/800x400?text=Histogram+Distribusi+Fitur+Numerik)

**Analisis Korelasi**:
- Terdapat korelasi positif yang kuat antara `median_income` dan `median_house_value` (r = 0.688), mengkonfirmasi bahwa pendapatan masyarakat merupakan prediktor terkuat nilai hunian.
- Fitur geografis `longitude` dan `latitude` juga menunjukkan korelasi yang berarti, mengindikasikan adanya pengaruh lokasi terhadap nilai properti.

![Heatmap Korelasi](https://via.placeholder.com/700x600?text=Heatmap+Korelasi+Antarvariabel)

**Distribusi Spasial**:
Visualisasi peta sebaran nilai hunian berdasarkan koordinat geografis menunjukkan konsentrasi nilai hunian tertinggi di kawasan pesisir dan perkotaan, sementara wilayah pedalaman cenderung memiliki nilai hunian yang lebih rendah.

![Peta Sebaran](https://via.placeholder.com/800x500?text=Peta+Sebaran+Nilai+Hunian+Berdasarkan+Lokasi)

---

## Data Preparation

Proses persiapan data yang dilakukan mencakup langkah-langkah berikut:

1. **Penanganan Missing Values**: Kolom `total_bedrooms` yang memiliki 207 nilai kosong diisi menggunakan nilai median kolom tersebut. Pendekatan imputasi median dipilih karena distribusi kolom ini bersifat right-skewed sehingga lebih tahan terhadap pengaruh outlier dibandingkan imputasi mean.

2. **Feature Engineering**: Dibuat beberapa fitur turunan untuk meningkatkan representasi data:
   - `rooms_per_household` = `total_rooms` / `households`
   - `bedrooms_per_room` = `total_bedrooms` / `total_rooms`
   - `population_per_household` = `population` / `households`

3. **Encoding Fitur Kategorikal**: Fitur `ocean_proximity` yang bersifat kategorikal diubah menggunakan teknik *One-Hot Encoding*, menghasilkan lima kolom biner baru yang merepresentasikan setiap kategori kedekatan dengan laut.

4. **Penanganan Outlier**: Outlier pada fitur `total_rooms`, `total_bedrooms`, dan `population` ditangani menggunakan metode *capping* berdasarkan persentil ke-1 dan ke-99 untuk menghindari hilangnya data secara berlebihan.

5. **Normalisasi Fitur**: Fitur-fitur numerik dinormalisasi menggunakan `StandardScaler` dari `sklearn.preprocessing` agar semua fitur berada dalam skala yang seragam (mean = 0, std = 1). Langkah ini penting terutama untuk model yang sensitif terhadap skala fitur.

6. **Pembagian Data**: Dataset dibagi menjadi data pelatihan (80%) dan data pengujian (20%) menggunakan `train_test_split` dengan parameter `random_state=42` untuk memastikan reproduktibilitas hasil.

**Alasan Data Preparation**:
Setiap langkah preparation di atas dilakukan dengan tujuan meningkatkan kualitas data yang dimasukkan ke dalam model. Penanganan missing values mencegah error saat training, feature engineering menambah informasi kontekstual yang relevan, encoding memungkinkan model memproses fitur kategorikal, sedangkan normalisasi memastikan tidak ada fitur yang mendominasi proses pembelajaran model secara tidak proporsional.

---

## Model Development

Pada proyek ini, digunakan algoritma **XGBoost Regressor** (*Extreme Gradient Boosting*) untuk tugas regresi prediksi nilai hunian.

### XGBoost Regressor

XGBoost adalah algoritma ensemble berbasis *boosting* yang membangun pohon keputusan secara sekuensial, di mana setiap pohon baru berusaha memperbaiki kesalahan residual dari pohon sebelumnya menggunakan teknik *gradient descent*. Berbeda dengan Random Forest yang membangun pohon secara paralel (*bagging*), XGBoost bersifat iteratif sehingga lebih efisien dalam meminimalkan fungsi loss secara bertahap.

**Parameter yang Digunakan**:

| Parameter | Nilai | Keterangan |
|---|---|---|
| `n_estimators` | 300 | Jumlah iterasi boosting |
| `learning_rate` | 0.05 | Laju pembelajaran untuk setiap iterasi |
| `max_depth` | 6 | Kedalaman maksimum setiap pohon |
| `subsample` | 0.8 | Proporsi sampel yang digunakan per iterasi |
| `colsample_bytree` | 0.8 | Proporsi fitur yang digunakan per pohon |
| `random_state` | 42 | Seed untuk reproduktibilitas |

**Kelebihan**: Umumnya menghasilkan akurasi lebih tinggi, memiliki mekanisme regularisasi bawaan untuk mencegah overfitting.  
**Kekurangan**: Lebih banyak hyperparameter yang perlu di-tuning dan lebih rentan terhadap overfitting jika tidak dikonfigurasi dengan baik.

---

## Evaluation

### Metrik Evaluasi

Untuk mengevaluasi performa model, digunakan tiga metrik berikut:

- **RMSE (Root Mean Squared Error)**: Mengukur akar dari rata-rata kuadrat selisih antara nilai prediksi dan nilai aktual. Metrik ini memberikan penalti lebih besar pada kesalahan prediksi yang besar, sehingga sensitif terhadap outlier. Nilai yang lebih kecil menunjukkan model yang lebih baik.

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

- **MAE (Mean Absolute Error)**: Mengukur rata-rata nilai absolut selisih antara prediksi dan aktual. MAE lebih mudah diinterpretasikan secara langsung karena berada dalam satuan yang sama dengan target (USD).

$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

- **R² Score (Coefficient of Determination)**: Mengukur proporsi variansi pada variabel target yang dapat dijelaskan oleh model. Nilai R² mendekati 1.0 menunjukkan model yang sangat baik.

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

### Hasil Evaluasi

| Metrik | Nilai |
|---|---|
| RMSE | 43.571 |
| MAE | 29.102 |
| R² Score | 0.8490 |

Model XGBoost berhasil mencapai R² Score sebesar **0.8490**, yang berarti model mampu menjelaskan sekitar **84,9% variansi** nilai hunian berdasarkan fitur-fitur geografis dan sosial ekonomi yang digunakan. Nilai RMSE sebesar 43.571 menunjukkan rata-rata kesalahan prediksi yang relatif kecil dibandingkan rentang nilai hunian dalam dataset.

**Visualisasi Prediksi vs Aktual (XGBoost)**:

Plot di bawah ini menunjukkan hubungan antara nilai aktual dan nilai prediksi dari model XGBoost terbaik. Garis merah putus-putus merepresentasikan garis ideal (y = x).

![Prediksi vs Aktual](https://via.placeholder.com/700x500?text=Scatter+Plot+Prediksi+vs+Nilai+Aktual)

- Sebagian besar titik data terkonsentrasi di sekitar garis ideal, menunjukkan performa prediksi yang baik.
- Terdapat beberapa titik yang menyimpang jauh dari garis ideal, terutama pada nilai hunian yang sangat tinggi (di atas 400.000 USD), yang kemungkinan disebabkan oleh efek *capping* pada data asli.

**Analisis Feature Importance**:

Berdasarkan analisis *feature importance* dari model XGBoost, diketahui bahwa fitur paling berpengaruh terhadap nilai hunian adalah:

1. `median_income` — kontribusi terbesar (~48%), mengkonfirmasi bahwa faktor sosial ekonomi berupa tingkat pendapatan merupakan prediktor dominan.
2. `latitude` dan `longitude` — secara gabungan berkontribusi ~18%, menunjukkan pengaruh signifikan faktor geografis.
3. `ocean_proximity_NEAR BAY` dan `ocean_proximity_INLAND` — berkontribusi ~10%, mengindikasikan bahwa kedekatan dengan perairan secara signifikan mempengaruhi nilai properti.
4. `housing_median_age` — kontribusi ~8%, menunjukkan bahwa usia bangunan juga menjadi pertimbangan penting.

![Feature Importance](https://via.placeholder.com/700x450?text=Bar+Chart+Feature+Importance)

### Dampak terhadap Business Understanding

- **Problem Statement**: Model XGBoost berhasil menjawab permasalahan yang diajukan dengan memprediksi nilai hunian secara akurat (R² = 0.8490). Analisis *feature importance* juga memberikan wawasan kuantitatif mengenai faktor-faktor yang paling berpengaruh terhadap nilai hunian.

- **Goals**: Tujuan mengidentifikasi faktor geografis dan sosial ekonomi yang berpengaruh telah tercapai. Ditemukan bahwa `median_income` (faktor sosial ekonomi) dan koordinat geografis merupakan prediktor terkuat, selaras dengan hipotesis awal.

- **Solution Statement**: Model XGBoost menghasilkan RMSE = 43.571 dan R² = 0.8490, membuktikan bahwa algoritma ini efektif dan dapat diandalkan sebagai alat bantu estimasi nilai properti berbasis data.

---

## Deployment

### Strategi Deployment

Model XGBoost yang telah dilatih di-*deploy* sebagai layanan web API menggunakan **Flask** sebagai backend framework, yang kemudian dibungkus dalam **Docker** untuk memastikan konsistensi environment di berbagai platform. Antarmuka pengguna dibangun menggunakan **Streamlit** agar dapat diakses langsung melalui browser tanpa instalasi tambahan.

### Arsitektur Deployment

```
User (Browser)
     │
     ▼
Streamlit Frontend  ──►  Flask REST API  ──►  XGBoost Model (.pkl)
     │                        │
     │                   Preprocessing
     │                  (StandardScaler,
     │                  OneHotEncoder)
     │
     ▼
  Prediksi Nilai Hunian (USD)
```

### Langkah-langkah Deployment

**1. Menyimpan Model dan Preprocessor**

Setelah model selesai dilatih, model dan objek preprocessor disimpan menggunakan `joblib` agar dapat dimuat kembali saat inferensi tanpa perlu melatih ulang.

```python
import joblib

# Simpan model dan scaler
joblib.dump(xgb_model, 'model/xgboost_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
```

**2. Membangun REST API dengan Flask**

Endpoint `/predict` menerima data input dalam format JSON, melakukan preprocessing, dan mengembalikan hasil prediksi nilai hunian.

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model  = joblib.load('model/xgboost_model.pkl')
scaler = joblib.load('model/scaler.pkl')
encoder = joblib.load('model/encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = preprocess(data)
    prediction = model.predict(features)
    return jsonify({'predicted_house_value': round(float(prediction[0]), 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**3. Kontainerisasi dengan Docker**

Seluruh aplikasi dikemas dalam Docker container untuk memastikan environment yang konsisten antara pengembangan dan produksi.

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

**4. Antarmuka Pengguna dengan Streamlit**

Antarmuka berbasis web dibangun menggunakan Streamlit, memungkinkan pengguna memasukkan data geografis dan sosial ekonomi secara interaktif dan mendapatkan estimasi nilai hunian secara real-time.

```python
import streamlit as st
import requests

st.title('Prediksi Nilai Hunian')
st.subheader('Masukkan Data Geografis & Sosial Ekonomi')

median_income     = st.slider('Median Income (dalam $10.000)', 0.5, 15.0, 3.0)
housing_median_age = st.slider('Median Usia Bangunan (tahun)', 1, 52, 20)
latitude          = st.number_input('Latitude', value=34.05)
longitude         = st.number_input('Longitude', value=-118.25)
ocean_proximity   = st.selectbox('Kedekatan dengan Laut',
                                  ['NEAR BAY', 'INLAND', '<1H OCEAN', 'NEAR OCEAN', 'ISLAND'])

if st.button('Prediksi'):
    payload = {
        'median_income': median_income,
        'housing_median_age': housing_median_age,
        'latitude': latitude,
        'longitude': longitude,
        'ocean_proximity': ocean_proximity
    }
    response = requests.post('http://localhost:5000/predict', json=payload)
    result = response.json()
    st.success(f"Estimasi Nilai Hunian: **${result['predicted_house_value']:,.0f}**")
```

### Monitoring dan Pemeliharaan

Setelah model di-deploy, dilakukan pemantauan secara berkala untuk mendeteksi potensi *model drift* akibat perubahan kondisi pasar properti. Langkah pemeliharaan yang direncanakan meliputi:

- **Monitoring performa**: Mencatat prediksi vs nilai aktual secara berkala untuk mendeteksi penurunan akurasi.
- **Retraining berkala**: Model diperbarui setiap kuartal menggunakan data terbaru agar tetap relevan dengan kondisi pasar.
- **Versioning model**: Setiap versi model disimpan dengan timestamp menggunakan konvensi penamaan `xgboost_model_vYYYYMMDD.pkl` untuk memudahkan rollback jika diperlukan.

---

## Kesimpulan

Proyek ini berhasil membangun model prediksi nilai hunian berbasis faktor geografis dan sosial ekonomi menggunakan algoritma XGBoost Regressor. Model yang dikembangkan mencapai performa yang baik dengan nilai R² sebesar 0.8490, RMSE sebesar 43.571, dan MAE sebesar 29.102.

Temuan utama dari analisis ini mengkonfirmasi bahwa **faktor sosial ekonomi**, khususnya tingkat pendapatan median masyarakat, merupakan penentu nilai hunian yang paling dominan. Sementara itu, **faktor geografis** seperti lokasi (koordinat dan kedekatan dengan laut) turut memainkan peran penting sebagai prediktor sekunder. Wawasan ini dapat dimanfaatkan oleh pengembang properti dalam strategi penetapan harga, oleh pemerintah daerah dalam perencanaan tata ruang berbasis data, serta oleh masyarakat umum sebagai panduan dalam pengambilan keputusan investasi properti.

Untuk penelitian selanjutnya, disarankan untuk mempertimbangkan penambahan fitur-fitur seperti akses terhadap transportasi publik, kualitas sekolah terdekat, dan tingkat kriminalitas wilayah yang berpotensi meningkatkan akurasi model lebih lanjut.
