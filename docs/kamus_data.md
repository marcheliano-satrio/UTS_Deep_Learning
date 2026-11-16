# Kamus data UTS Deep Learning

Semua kelompok menerima data yang sama. Label data uji tidak dibagikan.

## Tugas 1: klasifikasi_train.npz, klasifikasi_test.npz, klasifikasi_*_meta.csv

Satu sampel per unit produk dari pabrik komponen fiktif. Baris ke-i di berkas meta
memakai `indeks` = i pada array di berkas npz.

| Isi | Keterangan |
| --- | --- |
| `gambar` | Array uint8 (N, 64, 64), foto permukaan produk |
| `sensor` | Array float16 (N, 3, 128): suhu (C), getaran (relatif), arus (A) selama penekanan. NaN berarti data tidak terekam |
| id_unit | ID unit produk |
| lini | Lini produksi (A atau B) |
| shift | Shift produksi |
| id_mesin | Mesin penekan |
| batch_material | Batch bahan baku |
| kelas | Label: normal, gores, noda, retak, gelembung, las_dingin, kontaminasi. Hanya di data latih |

Label data latih berasal dari inspektur dan tidak dijamin bebas salah.

## Tugas 2: rul_train.csv.gz dan rul_test.csv.gz

Log per siklus untuk mesin press fiktif. Data latih berisi riwayat sampai mesin rusak.
Data uji berhenti pada titik acak sebelum rusak.

| Kolom | Keterangan |
| --- | --- |
| id_unit | ID mesin |
| siklus | Nomor siklus sejak mesin mulai dipakai |
| setel_1, setel_2, setel_3 | Pengaturan operasi pada siklus itu |
| s01-s21 | Bacaan sensor. NaN berarti bacaan hilang |

Label: sisa umur pakai (RUL) = jumlah siklus tersisa sampai rusak, dipotong pada 150
(sisa umur di atas 150 siklus dihitung 150). Di data latih, siklus terakhir tiap unit adalah
siklus saat rusak, jadi labelnya dihitung sendiri. Prediksi RUL dibuat pada siklus terakhir
tiap unit di data uji.

## Tugas 3: clustering.npz dan clustering_meta.csv

Sampel material dari laboratorium fiktif, tanpa label.

| Isi | Keterangan |
| --- | --- |
| `gambar` | Array uint8 (N, 64, 64), citra mikrostruktur dari mikroskop |
| `spektrum` | Array float16 (N, 128), spektrum komposisi |
| id_sampel | ID sampel |
| mikroskop | Mikroskop yang dipakai (M1 atau M2) |
