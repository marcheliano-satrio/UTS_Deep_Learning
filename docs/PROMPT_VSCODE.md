# Prompt pengerjaan di VS Code (urut, satu per satu)

Cara pakai: buka repo di VS Code, buka panel agen (Claude Code / Copilot Chat mode Agent),
lalu kirim prompt di bawah **satu per satu**. Setelah tiap prompt: baca hasilnya,
jalankan di Colab kalau diminta, baru lanjut ke prompt berikutnya (✅ = checkpoint).

Siapkan dulu di folder `docs/`:
- `soal_uts.md` (salin teks soal dari dosen)
- `kamus_data.md` (dari dataset)
- `ringkasan_data.txt` (hasil `00_setup_colab.ipynb`)

Prompt yang ditandai [Task 1/2/3] dikerjakan anggota yang memegang task itu,
di fork-nya sendiri, lalu dibuka PR.

---

## Tahap 0 — Orientasi (semua anggota)

**P0.1**
```
Baca CLAUDE.md, docs/soal_uts.md, docs/kamus_data.md, dan docs/ringkasan_data.txt.
Jangan tulis kode dulu. Buat ringkasan: (1) file data apa saja untuk tiap task,
kolom ID dan label, ukuran gambar, panjang sinyal/spektrum, format riwayat mesin;
(2) hal yang tidak jelas atau janggal di data; (3) rencana file kode per task.
Simpan ke docs/rencana.md lalu berhenti.
```
✅ Cek apakah ringkasannya cocok dengan kamus_data.md.

**P0.2**
```
Buat kode/buat_subset.py yang mengambil subset kecil (mis. 300 unit Task 1,
30 mesin Task 2, 300 sampel Task 3) dari --data ke --out dengan struktur folder
yang sama, supaya kode bisa diuji di CPU laptop. Berhenti setelah selesai.
```
✅ Jalankan di Colab, unduh hasilnya (zip), ekstrak ke `data_kecil/` di laptop.

**P0.3**
```
Buat kerangka kode/reproduce.py dengan argumen --data, --retrain, --task {1,2,3,all}.
Untuk sekarang hanya memanggil fungsi kosong per task dan mencetak versi library.
```

---

## Tahap 1 — Task 1: Klasifikasi multimodal [Task 1]

**P1.1 EDA**
```
Buat kode/task1_eda.py: contoh gambar dan plot sinyal per kelas, distribusi kelas,
persentase sensor hilang per kelas dan per jalur, perbedaan distribusi train vs test
per jalur produksi. Simpan gambar ke laporan_figs/task1/. Uji dengan --data data_kecil.
```
✅ Jalankan di Colab pada data penuh; kirim temuan ke anggota lain.

**P1.2 Dataset + baseline**
```
Buat task1_data.py (Dataset PyTorch gambar+sinyal, flag sensor_ada, normalisasi dari
train saja, split validasi stratified yang juga menyimpan info jalur) dan
task1_model.py (image encoder CNN kecil/ResNet18, signal encoder 1D-CNN).
Lalu task1_train.py dengan mode --modal {image,sensor,fusion}, --fusion {concat,late},
--smoke, class-weighted loss, metrik macro-F1 total + per kelas + per jalur.
Sertakan sanity check overfit satu batch (--overfit_batch).
```
✅ Jalankan `--smoke` di laptop, lalu overfit_batch dan baseline di Colab.

**P1.3 Fusion kedua + modality**
```
Tambahkan fusion cross-attention (atau pertahankan concat vs late kalau waktu sempit).
Buat skrip yang menjalankan image-only, sensor-only, dan tiap fusion lalu membuat
tabel F1 per kelas (CSV + gambar heatmap) untuk menjawab kelas mana butuh modality mana.
```

**P1.4 Label noise**
```
Buat task1_noise.py: prediksi out-of-fold 5-fold, tandai sampel dengan probabilitas label
yang sangat rendah sebagai kemungkinan salah label, simpan contoh gambarnya.
Lalu bandingkan training biasa vs label smoothing (dan opsional co-teaching).
```

**P1.5 Domain shift + missing modality**
```
Tambahkan: (a) evaluasi per jalur dan leave-one-line-out, (b) satu teknik penanganan
(augmentasi atau normalisasi per jalur), (c) modality dropout saat training
(sensor dikosongkan acak 25%) dan evaluasi validasi dengan sensor sengaja dihapus.
```

**P1.6 Final**
```
Latih model final dengan 3 seed, simpan learning curves, confusion matrix, mean±std
macro-F1. Buat prediksi_klasifikasi.csv (ensemble seed) dan jalankan cek_submission.py.
Sambungkan ke reproduce.py --task 1.
```

---

## Tahap 2 — Task 2: RUL [Task 2]

**P2.1 EDA + preprocessing**
```
Buat task2_eda.py: pengaruh operating settings ke sensor (scatter/cluster regime),
sensor konstan/tidak informatif, missing values, spikes, panjang riwayat train vs test.
Lalu task2_prep.py: hapus sensor tak informatif, normalisasi per regime operasi
(statistik dari train), penanganan spike kausal. Jelaskan pilihan normalisasi di docstring.
```

**P2.2 Label + window**
```
Buat label RUL train = min(150, siklus_terakhir - siklus). Buat sliding window
(panjang dapat diatur, default 30) tanpa data masa depan; riwayat pendek di-padding kiri
+ mask. Untuk test, ambil window yang berakhir di siklus terakhir. Split GroupKFold per mesin.
```

**P2.3 Model + baseline**
```
Buat task2_train.py dengan --model {gru,tcn,transformer} dan baseline non-DL
(Random Forest/LightGBM pada fitur window). Laporkan RMSE dan skor asimetris
(d<0: exp(-d/13)-1, d>=0: exp(d/10)-1) total dan per rentang RUL (0-50, 50-100, 100-150).
```

**P2.4 Ketidakpastian + final**
```
Ensemble 5 seed (dan/atau MC dropout) -> interval prediksi 90%. Analisis cakupan interval
di validasi dan aturan servis (mis. servis jika batas bawah < ambang). Buat prediksi_rul.csv
(clip 0-150), cek_submission, sambungkan ke reproduce.py --task 2.
```

---

## Tahap 3 — Task 3: Clustering [Task 3]

**P3.1 EDA**
```
Buat task3_eda.py: contoh gambar dengan distorsi (rotasi, skala, brightness, beda mikroskop)
dan spektrum (shift, baseline, spike). Simpan figur. Catat distorsi mana yang dominan.
```

**P3.2 Representasi tanpa label**
```
Buat task3_ssl.py: SimCLR untuk gambar (augmentasi meniru distorsi di EDA) dan untuk
spektrum dengan encoder 1D (augmentasi shift, baseline acak, spike, noise). Tambahkan
embedding pretrained (ResNet ImageNet, sitasi sumber) sebagai pembanding. Simpan embedding
ke .npy. Jelaskan alasan tiap augmentasi di docstring.
```

**P3.3 Fusion + clustering**
```
Buat task3_cluster.py: gabungkan embedding (normalisasi L2 + concat berbobot, opsional
kontrastif gambar-spektrum), bandingkan image-only, spectrum-only, fusion. Clustering
K-Means/GMM vs DEC (inisialisasi K-Means). Pilih k dengan silhouette, Davies-Bouldin,
dan BIC GMM, plus stabilitas antar seed.
```

**P3.4 Analisis + final**
```
UMAP embedding diwarnai cluster dan diwarnai atribut distorsi (mis. mikroskop) untuk cek
apakah cluster mengikuti pengganggu. Identifikasi cluster yang sulit dipisah. Buat
cluster.csv, cek_submission, sambungkan ke reproduce.py --task 3.
```

---

## Tahap 4 — Finalisasi (semua anggota)

**P4.1**
```
Periksa reproduce.py end-to-end: mode cepat harus selesai < 5 menit dari weights
tersimpan (untuk ujian lisan). Lengkapi kode/README.md: instalasi, lokasi data,
versi library, seed, cara retrain, lokasi/tautan weights.
```

**P4.2**
```
Kumpulkan semua angka dari runs/*/metrics.json menjadi tabel untuk laporan
(docs/tabel_laporan.md) dan daftar figur. Jangan mengarang angka; tandai yang belum ada.
```

**P4.3 (persiapan lisan)**
```
Buat docs/persiapan_lisan.md: 10 pertanyaan yang kemungkinan ditanya dosen per task
beserta jawaban singkat berdasarkan kode dan hasil kita, dan langkah reproduksi live.
```
