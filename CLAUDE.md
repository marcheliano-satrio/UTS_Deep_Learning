# CLAUDE.md — Aturan proyek UTS Deep Learning

Repo kelompok untuk UTS DL (3 task: klasifikasi multimodal, RUL, clustering multimodal).
Soal lengkap: `docs/soal_uts.md`. Deskripsi file data: `docs/kamus_data.md`.
Ringkasan isi data (hasil Colab): `docs/ringkasan_data.txt`. Urutan kerja: `docs/PROMPT_VSCODE.md`.

**Deadline: 24 Nov 2026 23:59.** Semua PR harus sudah di-merge ke `main` sebelum itu.

## Aturan wajib (dari soal — melanggar = kehilangan nilai)

1. **Model utama tiap task harus deep learning** (PyTorch). Metode klasik hanya sebagai baseline.
2. **Tidak boleh data eksternal.** Pretrained weights publik boleh, tapi sumbernya wajib disitasi
   (tulis di docstring dan di `kode/README.md`).
3. **Satu perintah mereproduksi semua angka laporan:** `python kode/reproduce.py --data <path> --task all`.
   Mode cepat (dari weights tersimpan) harus selesai < 5 menit untuk ujian lisan.
4. **Jangan pernah mengarang angka.** Setiap angka di laporan harus berasal dari file
   `runs/<nama_run>/metrics.json` yang dihasilkan kode. Kalau belum ada, tulis `TODO` / `belum dijalankan`.
5. **Nama file dan kolom prediksi harus persis:**
   - `prediksi_klasifikasi.csv` → `id_unit,kelas`; kelas ∈ {`normal`,`gores`,`noda`,`retak`,`gelembung`,`las_dingin`,`kontaminasi`}
   - `prediksi_rul.csv` → `id_unit,rul`; 0 ≤ rul ≤ 150
   - `cluster.csv` → `id_sampel,cluster`; cluster integer
   Ketiganya disimpan di root repo, satu baris per ID test/sampel.
6. **Catat seed dan versi library.** Setiap skrip training menerima `--seed` dan menyimpan
   `config.json` (argumen + versi torch/numpy/sklearn) di folder run-nya.

## Aturan anti-kebocoran data

- Statistik normalisasi (mean/std, per regime, per jalur) **dihitung dari train saja**.
- Task 1: split validasi stratified per kelas, simpan info jalur produksi untuk evaluasi per jalur.
- Task 2: split **per mesin** (GroupKFold), bukan per baris. Window hanya memakai siklus ≤ siklus saat ini
  (kausal). Penanganan spike juga kausal (tidak boleh pakai nilai masa depan).
- Task 3: tidak ada label. Jangan pakai atribut apa pun sebagai label untuk training.
- Data test hanya dipakai untuk prediksi akhir, tidak untuk memilih model atau hyperparameter.

## Cara kerja kode

- Semua kode di `kode/`. Path data selalu lewat argumen `--data` (jangan hard-code path Colab/laptop).
- Laptop (CPU) diuji dengan `--data data_kecil --smoke` (subset dari `kode/buat_subset.py`).
  Training penuh di Colab/Kaggle GPU.
- Output run → `runs/<task>_<nama>_seed<k>/` (metrics.json, config.json, kurva, weights).
  Figur laporan → `laporan_figs/task{1,2,3}/`.
- Skrip pendek dan bisa dijalankan sendiri (`argparse`, `if __name__ == "__main__":`).
- Docstring/komentar boleh bahasa Indonesia. Jelaskan *alasan* pilihan (normalisasi, augmentasi,
  arsitektur), karena semua anggota harus bisa menjelaskannya di ujian lisan.
- Jangan commit data, `data_kecil/`, atau file > 100 MB. Weights besar → tautan unduhan di README.

## Git & kontribusi

- Tiap anggota bekerja di fork sendiri, satu branch per fitur, lalu buka PR ke repo kelompok.
- Penggunaan AI dicatat di `kontribusi.md`: alat, bagian yang dibantu, dan cara verifikasinya
  (mis. smoke test, overfit satu batch, cek_submission, cek manual angka).
