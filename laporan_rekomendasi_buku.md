# Sistem Rekomendasi Buku - GoodBooks-10k

## Project Overview

Dalam era digital, pengguna dihadapkan pada begitu banyak pilihan bacaan, membuat proses menemukan buku yang sesuai menjadi tantangan tersendiri. Sistem rekomendasi hadir sebagai solusi untuk membantu pengguna menemukan konten yang relevan secara efisien.

Proyek ini bertujuan membangun sistem rekomendasi buku menggunakan dataset GoodBooks-10k dari Kaggle, yang mencakup 10.000+ buku dan hampir 1 juta rating dari pengguna. Sistem yang dikembangkan menggunakan pendekatan **Content-Based Filtering** dan **Collaborative Filtering**.

## Business Understanding

> Referensi: [Kaggle GoodBooks-10k Dataset](https://www.kaggle.com/datasets/zygmunt/goodbooks-10k)

### Problem Statements

1. Pengguna kesulitan menemukan buku yang sesuai dari ribuan pilihan.
2. Tidak semua pengguna memiliki histori interaksi, sehingga dibutuhkan pendekatan yang fleksibel.

### Goals

1. Membangun dua sistem rekomendasi: content-based dan collaborative filtering.
2. Memberikan rekomendasi buku personal untuk meningkatkan pengalaman pengguna.

### Solution Approach

- **Content-Based Filtering**: Memberikan rekomendasi berdasarkan kemiripan metadata buku seperti `title` dan `authors`.
- **Collaborative Filtering (SVD)**: Memberikan rekomendasi berdasarkan pola rating pengguna menggunakan matrix factorization.

## Data Understanding

Dataset terdiri dari dua file utama:

- `books.csv`: metadata buku
- `ratings.csv`: data rating pengguna

Jumlah data:

- **Books**: 10.000 baris
- **Ratings**: 980.000+ baris (sebelum pembersihan)

### Fitur dari books.csv:
- `id`: ID unik internal
- `book_id`: ID unik untuk setiap buku
- `best_book_id`: ID untuk versi terbaik dari buku tersebut
- `work_id`: ID unik untuk keseluruhan karya
- `books_count`: Jumlah edisi atau versi dari buku tersebut
- `isbn`: Kode ISBN 10 digit
- `isbn13`: Kode ISBN 13 digit
- `authors`: Nama penulis buku
- `original_publication_year`: Tahun terbit asli
- `original_title`: Judul asli buku
- `title`: Judul buku sesuai entri
- `language_code`: Kode bahasa buku
- `average_rating`: Rata-rata rating yang diterima buku
- `ratings_count`: Jumlah rating total yang diterima buku
- `work_ratings_count`: Jumlah rating berdasarkan karya
- `work_text_reviews_count`: Jumlah ulasan teks untuk karya tersebut
- `ratings_1` - `ratings_5`: Jumlah pengguna yang memberikan rating dari 1 hingga 5
- `image_url`: URL gambar sampul buku
- `small_image_url`: URL versi kecil dari gambar sampul buku

### Fitur dari ratings.csv:

- `user_id`: ID pengguna
- `book_id`: ID buku
- `rating`: Nilai rating dari 0-5

### Eksplorasi Awal:

- Beberapa kolom seperti `isbn`, `language_code`, dan `original_publication_year` memiliki nilai kosong.
- Terdapat 1.644 duplikat di `ratings.csv`.
- Distribusi rating bias ke arah nilai tinggi (4 dan 5).

## Data Preparation

Urutan data preparation:

1. **Menghapus kolom tidak relevan**:
   - `isbn`, `isbn13`, `original_title`, `language_code`, `image_url`, `small_image_url`
   - Alasan: Tidak digunakan dalam model rekomendasi.

2. **Menghapus duplikat** dari `ratings.csv`
   - Menghindari bias terhadap pengguna yang memberi rating sama lebih dari sekali.
   - 
3. **Menghapus baris** dengan missing value pada kolom 'original_publication_year'

4. **Gabungkan** `ratings` dan **`books`** berdasarkan `book_id`
   - Membuat basis data lengkap untuk evaluasi dan sistem CBF.

5. **Membuat kolom gabungan** `content_features`
   - Digabung dari `title` dan `authors` untuk digunakan dalam TF-IDF.

6. **TF-IDF Vectorization**
   - TF-IDF diterapkan pada kolom `content_features`
   - Mengubah teks menjadi vektor numerik.
   - Digunakan untuk menghitung kemiripan antar buku.

7. **Persiapan data untuk Collaborative Filtering (scikit-surprise)**
   - Format: `user_id`, `book_id`, `rating`

## Modeling and Results

### 1. Content-Based Filtering

- Cosine similarity digunakan untuk menghitung kemiripan antar buku.
- Output rekomendasi: 10 buku paling mirip dengan input judul.

**Contoh Output** – *Rekomendasi untuk “The Hobbit”*:
1. ['J.R.R. Tolkien 4-Book Boxed Set: The Hobbit and The Lord of the Rings'
2. 'The History of the Hobbit, Part One: Mr. Baggins',
3. 'The Children of Húrin',
4. 'The Hobbit: Graphic Novel',
5. 'Unfinished Tales of Númenor and Middle-Earth',
6. 'The Silmarillion (Middle-Earth Universe)',
7. 'The Two Towers (The Lord of the Rings, #2)',
8. 'The Return of the King (The Lord of the Rings, #3)',
9. 'The Fellowship of the Ring (The Lord of the Rings, #1)',
10. 'The Complete Guide to Middle-Earth']

### 2. Collaborative Filtering (SVD)

- Menggunakan library `scikit-surprise`.
- Data dibagi menjadi 80% training dan 20% testing.
- Model dibangun menggunakan `SVD()`, dilatih dan diuji menggunakan `train_test_split`.

### Simulasi Rekomendasi Buku untuk Pengguna dengan id_user : 123
Still Life with Woodpecker - Prediksi Rating: 4.72
Villa Incognito - Prediksi Rating: 4.64
The Beautiful and Damned - Prediksi Rating: 4.58
A People's History of the United States - Prediksi Rating: 4.55
The Taste of Home Cookbook - Prediksi Rating: 4.53

### Kelebihan dan Kekurangan:

| Metode              | Kelebihan                                                | Kekurangan                                                                   |
| ------------------- | -------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Content-Based       | Cocok untuk pengguna baru (cold-start), cepat diterapkan | Terbatas pada konten yang diketahui, tidak belajar dari rating pengguna lain |
| Collaborative (SVD) | Memberikan rekomendasi personal dari interaksi pengguna  | Tidak cocok untuk pengguna baru tanpa histori rating                         |

## Evaluation

### Metrik yang Digunakan

- **Precision@10**: Proporsi item relevan dari 10 rekomendasi teratas.
- **Recall@10**: Proporsi item relevan yang berhasil direkomendasikan dari seluruh item relevan.
- **RMSE dan MAE**: Metrik error untuk Collaborative Filtering.

### Formula Evaluasi:

- Precision@K = (Jumlah item relevan dalam top-K rekomendasi) / K
- Recall@K = (Jumlah item relevan dalam top-K rekomendasi) / (Total item relevan)

### Hasil Evaluasi:

| Algoritma           | Precision@10 | Recall@10 | RMSE   | MAE    |
|---------------------|--------------|-----------|--------|--------|
| Content-Based       | 0.1560       | 0.1044    | –      | –      |
| Collaborative (SVD) |      -       |    -      | 0.9066 | 0.7243 |

> Perhitungan untuk Content-Based dilakukan dengan membandingkan hasil rekomendasi terhadap buku yang pernah diberi rating tinggi oleh pengguna.

### Interpretasi:

- Collaborative Filtering (SVD) menghasilkan RMSE dan MAE yang cukup rendah (RMSE: 0.9072, MAE: 0.7246), menunjukkan akurasi prediksi rating yang baik.
- Content-Based Filtering menghasilkan Recall@10 sebesar 0.6521, yang berarti sekitar 65% dari buku-buku relevan (yang disukai pengguna) berhasil direkomendasikan. Precision@10-nya sebesar 0.1285, artinya dari 10 buku yang direkomendasikan, rata-rata 1–2 di antaranya benar-benar relevan untuk pengguna.

Setiap pendekatan memiliki kelebihan masing-masing.  
Content-Based Filtering unggul dalam mengatasi cold-start, sementara Collaborative Filtering efektif mempelajari preferensi pengguna aktif dan menghasilkan rekomendasi yang lebih akurat secara personal.
