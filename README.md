# Analisis Sentimen Imbalanced Data Ulasan Aplikasi Transjakarta (Complement Naive Bayes & Chi-Square)

Proyek *Natural Language Processing (NLP)* dan *Machine Learning* untuk mengklasifikasikan sentimen ulasan pengguna aplikasi Transjakarta di Google Play Store ke dalam dua kelas polaritas tegas: **Cibiran (Negatif)** dan **Pujian (Positif)**. 

Proyek ini berfokus pada penanganan masalah ketimpangan kelas (*imbalanced dataset* dengan rasio sekitar 18% Cibiran berbanding 82% Pujian) tanpa melakukan manipulasi data sintetis (*oversampling/undersampling*), melainkan melalui pendekatan matematis menggunakan **Complement Naive Bayes (CNB)** dan seleksi fitur statistik **Chi-Square ($\chi^2$)**.

---

## 📌 Latar Belakang & Solusi Teknis
Data ulasan aplikasi layanan publik umumnya didominasi oleh satu jenis sentimen. Pada algoritma klasifikasi standar, kondisi *imbalanced data* ini sering memicu bias sehingga model gagal mendeteksi keluhan pengguna (*kelas minoritas*).

Untuk mengatasi hal tersebut, proyek ini menerapkan strategi rekayasa berikut:
* **Polaritas Tegas (*Binary Filtering*):** Menghapus ulasan netral/ambigu (bintang 3 dan 4) agar model fokus mempelajari fitur kata yang memiliki sentimen kuat antara ulasan buruk (bintang 1–2) dan baik (bintang 5).
* **Proteksi Kata Negasi:** Mempertahankan kata-kata negasi penting (seperti *tidak*, *kurang*, *belum*, *jangan*, *bukan*) selama proses *stopword removal* agar makna kalimat tidak terbalik.
* **Normalisasi Slang Domain-Spesifik:** Menggunakan kamus normalisasi kustom untuk menangani singkatan bahasa Indonesia, istilah teknis aplikasi (*force close*, *bug*, *error*), hingga istilah transportasi (*tj*, *tije*, *halte*, *qris*).
* **Complement Naive Bayes (`alpha=0.1`):** Memanfaatkan estimasi probabilitas dari komplemen setiap kelas untuk mencegah bias terhadap kelas mayoritas.

---

## 📂 Struktur Dataset
Dataset berasal dari hasil *scraping* ulasan aplikasi Transjakarta di Google Play Store (total 2.090 data bersih setelah penyaringan, dibagi menjadi 1.672 data latih dan 418 data uji) dengan atribut utama:
* `content`: Teks ulasan asli yang ditulis oleh pengguna.
* `score`: Penilaian rating bintang dari pengguna (`1.0` hingga `5.0`).
* `at`: Waktu ulasan dikirimkan.
* `appVersion`: Versi aplikasi Transjakarta saat ulasan dibuat.
* `label_target` *(Fitur Hasil Rekayasa)*: Label biner hasil transformasi (`0` untuk kelas **Cibiran**, `1` untuk kelas **Pujian**).

---

## ⚙️ Alur Pemrosesan (Pipeline)
1. **Data Filtering & Labeling:** Menghapus nilai kosong pada kolom `content` dan `score`, menyaring rating bintang 3 dan 4, serta mengonversi label menjadi biner (`0` dan `1`).
2. **Text Preprocessing:**
   * *Case Folding* & pembersihan URL, tanda baca, serta angka menggunakan Regex (`re`).
   * Tokenisasi menggunakan `NLTK word_tokenize`.
   * Normalisasi kata tidak baku menggunakan kamus slang kustom (*custom dictionary*).
   * *Stopword Removal* gabungan bahasa Indonesia dan Inggris dengan pengecualian kata negasi.
   * *Stemming* bahasa Indonesia menggunakan pustaka **Sastrawi**.
3. **Data Splitting:** Membagi data menjadi **80% data latih** dan **20% data uji** (418 sampel) menggunakan *Stratified Split* (`random_state=42`) agar rasio kelas tetap terjaga.
4. **Ekstraksi & Seleksi Fitur:**
   * Pembobotan kata menggunakan **TF-IDF Vectorizer** (`max_features=5000`).
   * Penyaringan **500 fitur kata terbaik** (`k=500`) menggunakan uji statistik **Chi-Square ($\chi^2$)**.
5. **Pelatihan Model:** Melatih model **Complement Naive Bayes** dengan parameter *smoothing* `alpha=0.1`.

---

## 📊 Hasil & Evaluasi Model
Pengujian dilakukan pada **418 data uji** (*76 Cibiran* dan *342 Pujian*) dan menghasilkan performa yang sangat seimbang meskipun dataset memiliki ketimpangan kelas yang tinggi:

### 1. Confusion Matrix Biner

| Aktual \ Prediksi | Prediksi: Cibiran (0) | Prediksi: Pujian (1) |
| :--- | :---: | :---: |
| **Aktual: Cibiran (0)** | **73** *(True Negative)* | **3** *(False Positive)* |
| **Aktual: Pujian (1)** | **37** *(False Negative)* | **305** *(True Positive)* |

### 2. Laporan Performa Klasifikasi (*Classification Report*)

| Kelas / Metrik | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Cibiran (Kelas 0)** | 0.66 | **0.96** | 0.78 | 76 |
| **Pujian (Kelas 1)** | **0.99** | 0.89 | **0.94** | 342 |
| **Accuracy** | — | — | **0.90** | 418 |
| **Macro Avg** | 0.83 | 0.93 | 0.86 | 418 |
| **Weighted Avg** | 0.93 | 0.90 | 0.91 | 418 |

### 3. Ringkasan Insight Bisnis & Teknis
* **Skor Geometric Mean (G-Mean): `92.55%`** — Membuktikan bahwa model memiliki tingkat keseimbangan sensitivitas yang sangat tinggi dalam mengenali kedua kelas tanpa mengorbankan kelas minoritas.
* **Deteksi Keluhan Maksimal (*High Recall* pada Cibiran):** Model berhasil menangkap **96% (73 dari 76)** ulasan negatif/keluhan pengguna dengan tepat. Dalam konteks produk digital, kemampuan ini sangat krusial agar tim pengembang tidak melewatkan laporan *bug* atau kendala layanan dari pengguna.
* **Akurasi Keseluruhan: `90.43%`** — Sebanyak 378 dari 418 ulasan uji berhasil diklasifikasikan dengan benar.

---

## 🛠️ Pustaka & Teknologi
* **Bahasa Pemrograman:** Python 3
* **Manipulasi Data:** `pandas`, `numpy`
* **Pemrosesan Teks (NLP):** `nltk`, `Sastrawi`, `re`
* **Machine Learning & Metrik:** `scikit-learn`, `imbalanced-learn` (`imblearn`)

---

## 🚀 Cara Menjalankan Proyek
1. Klik tombol **"Open in Colab"** pada bagian atas file notebook (`.ipynb`) di repositori ini.
2. Pastikan pustaka tambahan sudah terpasang di lingkungan kerja Anda:
   ```bash
   pip install Sastrawi imbalanced-learn scikit-learn nltk pandas numpy
   ```
3. Unggah file dataset ulasan ke sesi kerja Anda dan sesuaikan jalur pembacaan file (`pd.read_csv(...)`).
4. Jalankan seluruh sel kode (*Run All*) untuk melihat hasil *preprocessing* hingga laporan evaluasi akhir.
