"""Ringkas isi folder data UTS menjadi satu file teks (docs/ringkasan_data.txt).

Dipakai sekali di Colab setelah data_uts_dl.zip diekstrak. Skrip ini sengaja generik
(tidak mengasumsikan struktur folder), karena tujuannya justru untuk MENGETAHUI struktur
data sebelum kode task ditulis. Hasilnya dibaca bersama kamus_data.md di prompt P0.1.

Yang diringkas:
- pohon folder: jumlah file per ekstensi per folder + 3 contoh nama file
- CSV / parquet: shape, kolom, dtype, jumlah null, jumlah nilai unik, value_counts
  untuk kolom kategorikal, dan 5 baris pertama
- .npy / .npz / .h5: shape, dtype, statistik dan jumlah NaN dari potongan awal
- gambar: ukuran dan mode dari sampel acak per folder
- .md / .txt / .json: potongan awal isi

Contoh:
    python kode/ringkas_data.py --data /content/data --out docs/ringkasan_data.txt
"""
import argparse
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
TABLE_EXT = {".csv", ".parquet", ".feather", ".tsv", ".gz"}  # .gz = .csv.gz
ARRAY_EXT = {".npy", ".npz", ".h5", ".hdf5"}
TEXT_EXT = {".md", ".txt", ".json", ".yaml", ".yml"}


def h(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def scan_tree(root):
    """Kembalikan {folder_relatif: Counter(ekstensi)} dan contoh nama file per folder."""
    counts = defaultdict(Counter)
    examples = defaultdict(list)
    sizes = defaultdict(int)
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel = os.path.relpath(dirpath, root)
        for f in sorted(filenames):
            ext = os.path.splitext(f)[1].lower() or "<tanpa_ekstensi>"
            counts[rel][ext] += 1
            if len(examples[rel]) < 3:
                examples[rel].append(f)
            try:
                sizes[rel] += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return counts, examples, sizes


def files_by_ext(root, exts, max_per_dir=None):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        picked = [f for f in sorted(filenames) if os.path.splitext(f)[1].lower() in exts]
        if max_per_dir is not None and len(picked) > max_per_dir:
            random.seed(0)
            picked = sorted(random.sample(picked, max_per_dir))
        out += [os.path.join(dirpath, f) for f in picked]
    return out


def summarize_table(path, root):
    import pandas as pd

    rel = os.path.relpath(path, root)
    ext = os.path.splitext(path)[1].lower()
    h(f"TABEL: {rel}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    try:
        if ext == ".parquet":
            df = pd.read_parquet(path)
        elif ext == ".feather":
            df = pd.read_feather(path)
        else:
            df = pd.read_csv(path, sep="\t" if ext == ".tsv" else ",", low_memory=False)
    except Exception as e:  # noqa: BLE001
        print(f"  gagal dibaca: {e!r}")
        return
    print(f"shape: {df.shape}")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "null": df.isna().sum(),
        "null_%": (df.isna().mean() * 100).round(2),
        "n_unik": df.nunique(dropna=True),
    })
    num = df.select_dtypes("number")
    if not num.empty:
        info["min"] = num.min()
        info["mean"] = num.mean().round(4)
        info["max"] = num.max()
    with_pd_opts(lambda: print(info.to_string()))
    print("\n5 baris pertama:")
    with_pd_opts(lambda: print(df.head().to_string()))
    for c in df.columns:
        nu = df[c].nunique(dropna=False)
        if 1 < nu <= 30:
            print(f"\nvalue_counts '{c}':")
            print(df[c].value_counts(dropna=False).to_string())
        elif nu == 1:
            print(f"\nkolom '{c}' KONSTAN: {df[c].iloc[0]!r}")
    # kolom yang tampak seperti ID: cek duplikat
    for c in df.columns:
        if "id" in c.lower():
            dup = df[c].duplicated().sum()
            print(f"\nkolom ID? '{c}': {df[c].nunique()} unik, {dup} baris duplikat, "
                  f"contoh {df[c].head(3).tolist()}")


def with_pd_opts(fn):
    import pandas as pd

    with pd.option_context("display.max_columns", 200, "display.width", 250,
                           "display.max_colwidth", 60):
        fn()


def describe_array(name, arr):
    import numpy as np

    print(f"  {name}: shape={arr.shape} dtype={arr.dtype}")
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.number):
        if arr.size:
            print(f"    contoh: {arr.reshape(-1)[:5]}")
        return
    sl = np.asarray(arr[: min(len(arr), 2000)], dtype=np.float64)
    n_nan = int(np.isnan(sl).sum())
    finite = sl[np.isfinite(sl)]
    if finite.size:
        print(f"    (2000 baris pertama) min={finite.min():.4g} mean={finite.mean():.4g} "
              f"std={finite.std():.4g} max={finite.max():.4g} NaN={n_nan}")
    else:
        print(f"    (2000 baris pertama) semua NaN/inf, NaN={n_nan}")


def summarize_array(path, root):
    import numpy as np

    rel = os.path.relpath(path, root)
    ext = os.path.splitext(path)[1].lower()
    h(f"ARRAY: {rel}  ({os.path.getsize(path) / 1e6:.1f} MB)")
    try:
        if ext == ".npy":
            describe_array("array", np.load(path, mmap_mode="r", allow_pickle=False))
        elif ext == ".npz":
            with np.load(path, allow_pickle=False) as z:
                for k in z.files:
                    describe_array(k, z[k])
        else:
            import h5py

            with h5py.File(path, "r") as f:
                def visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        describe_array(name, obj)
                f.visititems(visit)
    except Exception as e:  # noqa: BLE001
        print(f"  gagal dibaca: {e!r}")


def summarize_images(root, per_dir=20):
    from PIL import Image

    h("GAMBAR (sampel acak per folder)")
    by_dir = defaultdict(list)
    for p in files_by_ext(root, IMG_EXT, max_per_dir=per_dir):
        by_dir[os.path.relpath(os.path.dirname(p), root)].append(p)
    if not by_dir:
        print("  tidak ada file gambar lepas (mungkin gambar disimpan di dalam .npy/.h5)")
    for d, paths in sorted(by_dir.items()):
        meta = Counter()
        for p in paths:
            try:
                with Image.open(p) as im:
                    meta[(im.size, im.mode)] += 1
            except Exception as e:  # noqa: BLE001
                meta[("gagal", repr(e)[:40])] += 1
        print(f"  {d}: {dict(meta)}  (ukuran = lebar x tinggi)")


def summarize_text(path, root, n_chars=3000):
    rel = os.path.relpath(path, root)
    h(f"TEKS: {rel}")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except Exception as e:  # noqa: BLE001
        print(f"  gagal dibaca: {e!r}")
        return
    if path.lower().endswith(".json"):
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                print(f"  dict dengan {len(obj)} kunci, contoh kunci: {list(obj)[:10]}")
            elif isinstance(obj, list):
                print(f"  list dengan {len(obj)} elemen, elemen pertama: {str(obj[0])[:300]}")
        except Exception:  # noqa: BLE001
            pass
    print(txt[:n_chars] + ("\n  ...(dipotong)" if len(txt) > n_chars else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder hasil ekstrak data_uts_dl.zip")
    ap.add_argument("--out", default="ringkasan_data.txt")
    ap.add_argument("--max_tables", type=int, default=50)
    args = ap.parse_args()
    root = os.path.abspath(args.data)

    buf = io.StringIO()
    with redirect_stdout(buf):
        h(f"RINGKASAN DATA: {root}")
        print(f"python {sys.version.split()[0]}")
        counts, examples, sizes = scan_tree(root)
        h("POHON FOLDER (jumlah file per ekstensi)")
        for d in sorted(counts):
            print(f"{d}/  [{sizes[d] / 1e6:.1f} MB]  {dict(counts[d])}  contoh: {examples[d]}")

        for p in files_by_ext(root, TEXT_EXT, max_per_dir=5):
            summarize_text(p, root)
        # maks 10 tabel per folder: kalau sinyal disimpan 1 CSV per unit, cukup lihat contohnya
        for p in files_by_ext(root, TABLE_EXT, max_per_dir=10)[: args.max_tables]:
            summarize_table(p, root)
        for p in files_by_ext(root, ARRAY_EXT, max_per_dir=10):
            summarize_array(p, root)
        summarize_images(root)

    text = buf.getvalue()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text[:5000])
    print(f"\n[ok] ringkasan lengkap ({len(text)} karakter) disimpan ke {args.out}")


if __name__ == "__main__":
    main()
