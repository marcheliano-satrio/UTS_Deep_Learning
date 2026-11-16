"""Cek format 3 file prediksi sebelum push (dinilai otomatis, nama file & kolom harus persis).

Yang dicek per file:
- nama kolom persis dan urutannya (id_unit,kelas | id_unit,rul | id_sampel,cluster)
- tidak ada ID kosong/duplikat, tidak ada nilai kosong
- kelas ∈ 7 nama kelas persis; 0 <= rul <= 150; cluster integer
- (opsional) himpunan ID sama persis dengan ID test/sampel dari data: --ids_* menunjuk
  ke CSV berisi kolom ID yang benar. Diisi otomatis oleh reproduce.py setelah struktur
  data diketahui.

Contoh:
    python kode/cek_submission.py                # cek file di root repo
    python kode/cek_submission.py --file prediksi_rul.csv
    python kode/cek_submission.py --data data_uts_dl   # + cek himpunan ID sama persis dengan data
Exit code 1 kalau ada error, supaya bisa dipakai di skrip.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrik import KELAS  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEK = {
    "prediksi_klasifikasi.csv": ("id_unit", "kelas"),
    "prediksi_rul.csv": ("id_unit", "rul"),
    "cluster.csv": ("id_sampel", "cluster"),
}


def cek_file(path, ids_benar=None):
    nama = os.path.basename(path)
    err, info = [], []
    if nama not in SPEK:
        return [f"nama file '{nama}' bukan salah satu dari {list(SPEK)}"], info
    if not os.path.exists(path):
        return [f"file tidak ada: {path}"], info
    kol_id, kol_nilai = SPEK[nama]
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline().strip()
    if header != f"{kol_id},{kol_nilai}":
        err.append(f"header harus '{kol_id},{kol_nilai}', ditemukan '{header}'")
    df = pd.read_csv(path, dtype={kol_id: str}, keep_default_na=False)
    if list(df.columns) != [kol_id, kol_nilai]:
        return err + [f"kolom {list(df.columns)}"], info
    info.append(f"{len(df)} baris, contoh: {df.iloc[0].tolist() if len(df) else '-'}")
    ids = df[kol_id].astype(str).str.strip()
    if (ids == "").any():
        err.append(f"{(ids == '').sum()} ID kosong")
    if ids.duplicated().any():
        err.append(f"{ids.duplicated().sum()} ID duplikat, contoh {ids[ids.duplicated()].head(3).tolist()}")
    v = df[kol_nilai]
    if nama == "prediksi_klasifikasi.csv":
        salah = ~v.astype(str).isin(KELAS)
        if salah.any():
            err.append(f"{salah.sum()} kelas tidak valid, contoh {v[salah].unique()[:5].tolist()}")
        info.append(f"distribusi: {v.value_counts().to_dict()}")
    elif nama == "prediksi_rul.csv":
        x = pd.to_numeric(v, errors="coerce")
        if x.isna().any():
            err.append(f"{x.isna().sum()} rul bukan angka")
        if ((x < 0) | (x > 150)).any():
            err.append(f"{((x < 0) | (x > 150)).sum()} rul di luar [0,150]")
        info.append(f"rul min={x.min():.2f} mean={x.mean():.2f} max={x.max():.2f}")
    else:
        x = pd.to_numeric(v, errors="coerce")
        if x.isna().any() or not np.all(np.mod(x.dropna(), 1) == 0) or v.astype(str).str.contains(r"\.").any():
            err.append("cluster harus integer (tanpa desimal)")
        else:
            info.append(f"{x.nunique()} cluster, ukuran terkecil={x.value_counts().min()}, "
                        f"terbesar={x.value_counts().max()}")
    if ids_benar is not None:
        ids_benar = set(map(str, ids_benar))
        kurang, lebih = ids_benar - set(ids), set(ids) - ids_benar
        if kurang:
            err.append(f"{len(kurang)} ID test tidak ada di prediksi, contoh {sorted(kurang)[:3]}")
        if lebih:
            err.append(f"{len(lebih)} ID asing di prediksi, contoh {sorted(lebih)[:3]}")
    return err, info


def ids_dari_data(data_dir):
    """ID yang benar untuk tiap file prediksi, langsung dari data (bukan label, aman dipakai)."""
    return {
        "prediksi_klasifikasi.csv": pd.read_csv(os.path.join(data_dir, "klasifikasi_test_meta.csv"),
                                                dtype={"id_unit": str})["id_unit"],
        "prediksi_rul.csv": pd.read_csv(os.path.join(data_dir, "rul_test.csv.gz"), usecols=["id_unit"],
                                        dtype={"id_unit": str})["id_unit"].unique(),
        "cluster.csv": pd.read_csv(os.path.join(data_dir, "clustering_meta.csv"),
                                   dtype={"id_sampel": str})["id_sampel"],
    }


def cek(files=None, ids=None):
    """files: list path; ids: dict nama_file -> iterable ID benar. Return True kalau lolos."""
    files = files or [os.path.join(ROOT, n) for n in SPEK]
    ids = ids or {}
    lolos = True
    for p in files:
        err, info = cek_file(p, ids.get(os.path.basename(p)))
        status = "OK " if not err else "GAGAL"
        print(f"[{status}] {os.path.basename(p)}")
        for i in info:
            print(f"     {i}")
        for e in err:
            print(f"   ! {e}")
        lolos &= not err
    return lolos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", nargs="*", help="default: ketiga file di root repo")
    ap.add_argument("--data", default=None, help="folder data: ID test/sampel diambil otomatis dari sini")
    for n, (kol_id, _) in SPEK.items():
        key = n.split(".")[0]
        ap.add_argument(f"--ids_{key}", help=f"CSV berisi kolom {kol_id} yang benar untuk {n}")
    a = ap.parse_args()
    ids = ids_dari_data(a.data) if a.data else {}
    for n, (kol_id, _) in SPEK.items():
        p = getattr(a, f"ids_{n.split('.')[0]}")
        if p:
            ids[n] = pd.read_csv(p, dtype={kol_id: str})[kol_id]
    sys.exit(0 if cek(a.file, ids) else 1)


if __name__ == "__main__":
    main()
