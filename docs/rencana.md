# Rencana kerja & ringkasan data (P0.1)

Sumber: `docs/kamus_data.md`, `docs/ringkasan_data.txt`, dan skrip EDA (`kode/task*_eda.py`,
angka lengkap di `runs/t*_eda/metrics.json`).

## 1. Isi data per task

| Task | File | Isi | ID | Label |
|---|---|---|---|---|
| 1 | `klasifikasi_train.npz` + `_meta.csv` | 48.000 unit; `gambar` uint8 (N,64,64); `sensor` float16 (N,3,128) = suhu, getaran, arus | `id_unit` (U…), `indeks` = baris di npz | `kelas` (7 kelas), hanya train |
| 1 | `klasifikasi_test.npz` + `_meta.csv` | 16.000 unit, struktur sama tanpa `kelas` | `id_unit` | tersembunyi |
| 2 | `rul_train.csv.gz` | 900 mesin, 254.174 baris; `siklus`, `setel_1..3`, `s01..s21`; riwayat sampai rusak | `id_unit` (int) | dihitung: min(150, siklus_akhir − siklus) |
| 2 | `rul_test.csv.gz` | 300 mesin, 51.692 baris, terpotong di titik acak | `id_unit` | tersembunyi (RUL di siklus terakhir) |
| 3 | `clustering.npz` + `_meta.csv` | 36.000 sampel; `gambar` uint8 (N,64,64); `spektrum` float16 (N,128) | `id_sampel` (S…) | tidak ada; `mikroskop` M1/M2 = atribut pengganggu |

Meta Task 1 juga punya `lini` (A/B), `shift`, `id_mesin` (M01–M12), `batch_material` (150 batch).

## 2. Hal yang janggal / penting

**Task 1**
- Kelas tidak seimbang: normal 49,8%, kontaminasi hanya 4,9% → loss berbobot kelas, metrik macro-F1.
- **Domain shift besar**: lini B 25% di train tetapi 60% di test. Mesin M01–M06 = lini A,
  M07–M12 = lini B. Gambar lini B lebih gelap (kecerahan 112 vs 140) dan kontrasnya lebih tinggi,
  bertekstur garis diagonal; suhu lini B ±2 °C lebih tinggi. Distribusi kelas per lini sama.
- Sensor hilang **per kanal secara utuh** (tidak pernah sebagian): ~6% per kanal, ~4,9% unit
  kehilangan ketiganya; merata antar kelas dan lini, juga di test.
- Dari rata-rata sinyal per kelas: `las_dingin` terlihat jelas di arus (plateau lebih rendah),
  `gelembung` sedikit di suhu, `retak` sedikit di energi getaran; kelas lain terutama dari gambar.
- Label dari inspektur, mungkin salah → cek dengan prediksi out-of-fold.

**Task 2**
- `setel_1` hanya di sekitar {0, 10, 20, 25, 35, 42} → 6 regime operasi (setel_2/3 ikut ditentukan).
  Level sensor didominasi regime → normalisasi per regime.
- 7 sensor (s01, s05, s06, s10, s16, s18, s19) tidak berkorelasi dengan RUL setelah normalisasi → dibuang.
- NaN ~2% per sensor, acak (35% baris punya ≥1 NaN). Spike tunggal kecil (~0,15% nilai).
- Panjang riwayat train 133–778 (rata-rata 282); test 50–455 (rata-rata 172) karena terpotong.

**Task 3**
- Orientasi tekstur merata 0–180° → rotasi adalah distorsi, bukan ciri material.
- M2 lebih terang (146 vs 128) dan lebih tajam; M1 lebih blur. 60% sampel dari M1.
- Spektrum: 2–3 puncak di posisi bervariasi, baseline miring/lengkung, ~22% punya spike tunggal.
- Jumlah jenis material tidak diketahui → beberapa kriteria k.

## 3. File kode

| File | Isi |
|---|---|
| `common.py`, `metrik.py`, `cek_submission.py` | utilitas, metrik resmi, cek format prediksi |
| `ringkas_data.py`, `buat_subset.py` | ringkasan data, subset kecil untuk uji CPU |
| `task1_eda.py` | EDA Task 1 |
| `task1_data.py`, `task1_model.py`, `task1_train.py` | data GPU, encoder + 3 fusion, training/evaluasi |
| `task1_noise.py` | deteksi salah label (OOF 5-fold) |
| `task1_eksperimen.py` | grup eksperimen (modality, noise, domain, missing, final) + tabel/figur |
| `task2_eda.py`, `task2_prep.py` | EDA, preprocessing, label, window |
| `task2_model.py`, `task2_train.py` | GRU/TCN/Transformer + LightGBM, CV per mesin |
| `task2_eksperimen.py` | perbandingan model, ketidakpastian, aturan servis, final |
| `task3_eda.py`, `task3_ssl.py`, `task3_cluster.py` | EDA, SimCLR/joint/pretrained, clustering + analisis |
| `reproduce.py` | satu perintah untuk semua angka & file prediksi |
