# Laporan Proyek Machine Learning - Salma Nurfauziah

## Domain Proyek

### Latar Belakang
Konsumsi makanan adalah sektor penting yang mendukung perekonomian Indonesia, namun rendahnya kualitas nutrisi dalam makanan yang dikonsumsi masyarakat Indonesia menjadi masalah besar dalam menjaga kesehatan. Berdasarkan data, masyarakat Indonesia lebih cenderung mengonsumsi makanan instan yang praktis dan cepat, namun dengan kandungan gizi yang seringkali tidak seimbang. Oleh karena itu, penting untuk melakukan analisis terhadap asupan gizi masyarakat Indonesia agar mereka bisa lebih memilih makanan yang sehat dan bergizi.

### Masalah yang Harus Diselesaikan:
Berdasarkan penelitian yang menunjukkan bahwa sekitar 68% populasi Indonesia tidak mampu memenuhi kebutuhan gizi harian yang seimbang, permasalahan ini perlu diselesaikan dengan memberikan pemahaman yang lebih baik tentang kandungan gizi makanan yang dikonsumsi masyarakat, serta membantu mereka dalam memilih makanan yang bergizi dan seimbang.

  Format Referensi: [Rendahnya Asupan Nutrisi Masyarakat Indonesia](https://www.kompas.id/baca/riset/2023/01/31/rendahnya-asupan-nutrisi-masyarakat-indonesia) 

## Business Understanding

### Problem Statements

Bagaimana cara mengidentifikasi dan menganalisis kandungan gizi dari makanan yang dikonsumsi oleh masyarakat Indonesia menggunakan dataset yang ada, dan memberikan rekomendasi makanan yang lebih bergizi berdasarkan analisis tersebut?

### Goals

- Mengembangkan sistem atau model yang dapat memberikan pemahaman tentang kandungan gizi dalam berbagai makanan yang dikonsumsi oleh masyarakat Indonesia.
- Menyediakan saran makanan yang bergizi untuk meningkatkan kualitas pola makan masyarakat.

    ### Solution statements
    - Pendekatan pertama: Menggunakan algoritma regresi untuk memprediksi kandungan gizi (seperti kalori, protein, lemak, karbohidrat) dari berbagai jenis makanan dalam dataset.
    - Pendekatan kedua: Membangun model klasifikasi yang mengategorikan makanan menjadi kategori sehat atau tidak sehat berdasarkan profil gizi mereka, dan memberikan rekomendasi berdasarkan hasil klasifikasi tersebut.

  Evaluasi dari solusi ini akan menggunakan metrik seperti Akurasi untuk klasifikasi dan RMSE (Root Mean Squared Error) untuk regresi, untuk memastikan bahwa model dapat memberikan rekomendasi yang relevan dan akurat.
      
## Data Understanding
Dataset ini terdiri dari informasi mengenai lebih dari seribu jenis makanan dan minuman khas Indonesia, dengan kolom-kolom seperti kalori, protein, lemak, karbohidrat, dan nama makanan. 

Sumber dataset: [Kaggle - Indonesian Food and Drink Nutrition Dataset](https://www.kaggle.com/datasets/anasfikrihanif/indonesian-food-and-drink-nutrition-dataset) 

**Informasi Dataset**:
- Jumlah data: Dataset ini terdiri dari 1346 baris dan 6 kolom.
- Kondisi data:
    - Missing values: Hasil pengecekan menunjukkan bahwa tidak ditemukan missing values pada dataset, berdasarkan output dari isnull().sum().
    - Duplikat: Tidak ditemukan data duplikat.
    - Outlier: Tidak ada langkah eksplisit untuk mengidentifikasi atau menangani outlier dalam kode, namun distribusi data untuk fitur kalori, protein, lemak, dan karbohidrat telah divisualisasikan menggunakan histogram untuk memberikan gambaran tentang persebaran nilai. Tidak ada indikasi penanganan khusus terhadap nilai ekstrim dalam dataset.
      
**Fitur pada Indonesian Food and Drink Nutrition Dataset adalah sebagai berikut**:
- Id: digunakan sebagai identifikasi unik untuk setiap baris data
- Calories: Kalori per porsi makanan.
- Proteins: Kandungan protein per porsi makanan.
- Fat: Kandungan lemak per porsi makanan.
- Carbohydrate: Kandungan karbohidrat per porsi makanan.
- Name: Nama makanan.
- Image: URL gambar makanan yang memberikan gambaran visual dari makanan tersebut.

### Exploratory Data Analysis (EDA):
Sebelum melanjutkan ke tahap modeling, kita akan melakukan eksplorasi data untuk memahami distribusi kalori, protein, lemak, dan karbohidrat pada dataset ini. Visualisasi seperti histogram dan box plot akan membantu dalam mengeksplorasi pola dan potensi outlier dalam data.

**Visualisasi**:
- Distribusi Kalori: Kalori cenderung memiliki distribusi yang lebih terpusat dengan beberapa nilai ekstrim pada sisi kanan, yang menunjukkan adanya makanan dengan kandungan kalori yang sangat tinggi.
- Distribusi Protein: Mayoritas makanan memiliki kandungan protein yang lebih rendah, namun ada beberapa yang memiliki kadar protein yang cukup tinggi.
- Distribusi Lemak: Lemak memiliki distribusi yang sangat bervariasi, dengan sebagian besar makanan memiliki kandungan lemak yang relatif rendah.
- Distribusi Karbohidrat: Sebagian besar makanan memiliki kandungan karbohidrat yang lebih tinggi, menunjukkan bahwa karbohidrat adalah komponen utama dalam makanan ini.

![nut](https://github.com/user-attachments/assets/ba8547ae-0bf3-48e9-b33d-e7103c7b5aea)

## Data Preparation
Proses data preparation yang dilakukan pada dataset ini mencakup langkah-langkah berikut:
- Pembersihan Data: Pada langkah ini, kode yang digunakan mengecek apakah ada nilai yang hilang (missing values) menggunakan nutrition_data.isnull().sum(). Hasil dari pengecekan ini menunjukkan bahwa tidak ada missing values pada dataset. Oleh karena itu, tidak ada pembersihan data yang diperlukan.
- Normalisasi: Fitur-fitur numerik seperti kalori, protein, lemak, dan karbohidrat dinormalisasi menggunakan StandardScaler dari sklearn.preprocessing. Normalisasi ini memastikan bahwa data berada dalam skala yang seragam dengan rata-rata 0 dan standar deviasi 1. Proses ini penting untuk menghindari bias akibat perbedaan skala antar fitur saat diterapkan dalam model machine learning.
- Pembagian Data: Dataset dibagi menjadi data pelatihan dan data pengujian menggunakan teknik train-test split dengan komposisi 80% untuk data pelatihan dan 20% untuk data pengujian. Hal ini dilakukan untuk memastikan model dapat diuji dengan data yang belum pernah dilihat sebelumnya.

**Alasan Data Preparation**:
Langkah-langkah ini penting untuk memastikan data siap digunakan dalam model machine learning. Pembersihan dan normalisasi data memastikan model bekerja dengan data yang akurat dan konsisten, serta tidak terpengaruh oleh masalah multikolinearitas atau skala yang berbeda pada fitur.

## Model Development
Pada bagian ini, saya menggunakan algoritma Linear Regression untuk memprediksi kandungan kalori berdasarkan fitur protein, lemak, dan karbohidrat. Algoritma ini memodelkan hubungan linier antara variabel independen (protein, lemak, karbohidrat) dan target (kalori).

**Parameter yang Digunakan**:
- fit_intercept: True (parameter default) - Menandakan bahwa model akan mencari nilai intercept dalam persamaan regresi.
- normalize: False (parameter default) - Data tidak dinormalisasi secara internal oleh model karena telah dinormalisasi sebelumnya.

## Evaluation
Untuk mengevaluasi model, kita menggunakan beberapa metrik untuk mengukur kinerja model dalam memberikan rekomendasi yang relevan dan akurat.

**Metrik Evaluasi yang Digunakan**:
- Regresi Linier: Metrik yang digunakan adalah MSE (Mean Squared Error) untuk mengukur seberapa akurat prediksi kandungan gizi (seperti kalori) oleh model berdasarkan fitur lainnya. MSE memberikan gambaran tentang rata-rata kuadrat kesalahan antara nilai prediksi dan nilai aktual, dengan nilai yang lebih kecil menunjukkan model yang lebih akurat.

**Hasil Evaluasi**:
Plot Prediksi vs Actual di bawah menunjukkan hubungan antara nilai aktual dan nilai prediksi dari model regresi linier. Garis merah putus-putus merepresentasikan garis ideal di mana nilai prediksi sama dengan nilai aktual (y = x).

![pre](https://github.com/user-attachments/assets/c83f8c31-e837-4c44-bed8-ce20ef2c0001)

- Sebagian besar data terdistribusi di sekitar garis ideal, menunjukkan bahwa model regresi memiliki performa yang baik dalam memprediksi kandungan kalori.
- Terdapat beberapa titik yang jauh dari garis ideal, yang mengindikasikan adanya kesalahan prediksi pada beberapa sampel. Namun, jumlah titik tersebut relatif kecil.

Visualisasi ini mendukung hasil evaluasi numerik, yaitu MSE sebesar 0.157, yang menunjukkan bahwa model memiliki kesalahan prediksi yang kecil dan performa yang cukup akurat dalam memprediksi kandungan kalori makanan.

**Dampak terhadap Business Understanding**:
- Problem Statement: Model ini berhasil menjawab problem statement mengenai prediksi kandungan gizi makanan dan memberikan wawasan tentang kualitas gizi makanan yang dikonsumsi oleh masyarakat Indonesia.
- Goals: Model ini mencapai tujuan untuk memberikan pemahaman yang lebih baik tentang kandungan gizi dalam berbagai makanan yang dikonsumsi oleh masyarakat Indonesia dan dapat digunakan untuk memberikan rekomendasi makanan yang lebih sehat.
- Solution Statement: Algoritma regresi linier memberikan hasil yang baik dalam memprediksi kalori, protein, lemak, dan karbohidrat berdasarkan fitur lainnya. Model klasifikasi juga memiliki potensi untuk memberikan rekomendasi makanan yang lebih bergizi, yang dapat digunakan untuk meningkatkan kualitas pola makan masyarakat Indonesia.

Evaluasi ini menunjukkan bahwa solusi yang diberikan cukup efektif, namun perlu dilakukan lebih banyak pengujian dan perbaikan agar lebih akurat dalam memberikan rekomendasi yang lebih mendalam terkait pola makan yang sehat.

## Kesimpulan 
Laporan ini menunjukkan pentingnya analisis data untuk membantu masyarakat Indonesia dalam memilih makanan yang bergizi dan seimbang. Dengan menggunakan dataset Indonesian Food and Drink Nutrition Dataset, model ini bertujuan untuk memberikan rekomendasi yang dapat meningkatkan kualitas konsumsi gizi masyarakat, serta mengurangi prevalensi kekurangan gizi yang masih tinggi di Indonesia.