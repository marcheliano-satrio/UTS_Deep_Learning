"""Buat subset kecil data UTS dengan struktur file yang sama, untuk uji kode di CPU.

Default: 300 unit Task 1 (train) + 100 unit test, 30 mesin Task 2 (train) + 10 test,
300 sampel Task 3. Kolom `indeks` di meta ditulis ulang agar tetap = posisi di npz.

    python kode/buat_subset.py --data data_uts_dl --out data_kecil
"""
import argparse
import os
import shutil

import numpy as np
import pandas as pd


def subset_npz(data, out, nama, n, seed, kolom_strat=None):
    meta = pd.read_csv(os.path.join(data, f"{nama}_meta.csv"))
    rng = np.random.RandomState(seed)
    if kolom_strat and kolom_strat in meta:
        idx = meta.groupby(kolom_strat, group_keys=False).apply(
            lambda g: g.sample(max(1, round(n * len(g) / len(meta))), random_state=seed)).index.values
    else:
        idx = rng.choice(len(meta), min(n, len(meta)), replace=False)
    idx = np.sort(idx)
    z = np.load(os.path.join(data, f"{nama}.npz"))
    np.savez_compressed(os.path.join(out, f"{nama}.npz"), **{k: z[k][idx] for k in z.files})
    m = meta.iloc[idx].copy()
    m["indeks"] = np.arange(len(m))
    m.to_csv(os.path.join(out, f"{nama}_meta.csv"), index=False)
    print(f"{nama}: {len(m)} sampel")


def subset_rul(data, out, split, n, seed):
    df = pd.read_csv(os.path.join(data, f"rul_{split}.csv.gz"))
    unit = np.random.RandomState(seed).choice(df.id_unit.unique(), n, replace=False)
    d = df[df.id_unit.isin(unit)]
    d.to_csv(os.path.join(out, f"rul_{split}.csv.gz"), index=False)
    print(f"rul_{split}: {d.id_unit.nunique()} mesin, {len(d)} baris")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_uts_dl")
    ap.add_argument("--out", default="data_kecil")
    ap.add_argument("--n1", type=int, default=300)
    ap.add_argument("--n2", type=int, default=30)
    ap.add_argument("--n3", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    subset_npz(a.data, a.out, "klasifikasi_train", a.n1, a.seed, "kelas")
    subset_npz(a.data, a.out, "klasifikasi_test", a.n1 // 3, a.seed)
    subset_rul(a.data, a.out, "train", a.n2, a.seed)
    subset_rul(a.data, a.out, "test", max(5, a.n2 // 3), a.seed)
    subset_npz(a.data, a.out, "clustering", a.n3, a.seed)
    shutil.copy(os.path.join(a.data, "kamus_data.md"), os.path.join(a.out, "kamus_data.md"))


if __name__ == "__main__":
    main()
