"""Metrik evaluasi yang sama dengan penilaian dosen (docs/soal_uts.md).

- Task 1: macro-F1 atas 7 kelas (+ F1 per kelas, per jalur produksi).
- Task 2: RMSE dan skor asimetris s = sum(exp(-d/13)-1 jika d<0, exp(d/10)-1 jika d>=0),
  d = prediksi - aktual. Prediksi terlambat (d>0, terlalu optimis) dihukum lebih berat.
  Dilaporkan total dan per rentang RUL aktual (0-50, 50-100, 100-150).
- Task 3: ARI hanya bisa dihitung dosen (label tersembunyi); untuk kita: silhouette, DB, BIC.
"""
import numpy as np

KELAS = ["normal", "gores", "noda", "retak", "gelembung", "las_dingin", "kontaminasi"]
RUL_CAP = 150.0
RENTANG_RUL = [(0, 50), (50, 100), (100, 150)]


# ---------------------------------------------------------------- Task 1
def f1_per_kelas(y_true, y_pred, n_kelas=len(KELAS), bobot=None):
    """F1 per kelas dari label integer. Kelas tanpa sampel & tanpa prediksi -> NaN.

    bobot (opsional): bobot per sampel, mis. untuk meniru proporsi lini di data test.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    w = np.ones(len(y_true)) if bobot is None else np.asarray(bobot, float)
    out = np.full(n_kelas, np.nan)
    for k in range(n_kelas):
        tp = np.sum(w * ((y_pred == k) & (y_true == k)))
        fp = np.sum(w * ((y_pred == k) & (y_true != k)))
        fn = np.sum(w * ((y_pred != k) & (y_true == k)))
        if tp + fp + fn > 0:
            out[k] = 2 * tp / (2 * tp + fp + fn)
    return out


def macro_f1(y_true, y_pred, n_kelas=len(KELAS), bobot=None):
    """Macro-F1 seperti sklearn f1_score(average='macro') atas kelas yang muncul."""
    return float(np.nanmean(f1_per_kelas(y_true, y_pred, n_kelas, bobot)))


def ringkasan_klasifikasi(y_true, y_pred, grup=None):
    """dict: macro_f1, f1 per kelas, dan (opsional) macro_f1 per grup (mis. jalur)."""
    f1 = f1_per_kelas(y_true, y_pred)
    hasil = {
        "macro_f1": float(np.nanmean(f1)),
        "akurasi": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))),
        "f1_per_kelas": {KELAS[k]: (None if np.isnan(v) else float(v)) for k, v in enumerate(f1)},
    }
    if grup is not None:
        grup = np.asarray(grup)
        hasil["macro_f1_per_grup"] = {
            str(g): macro_f1(np.asarray(y_true)[grup == g], np.asarray(y_pred)[grup == g])
            for g in sorted(np.unique(grup), key=str)
        }
    return hasil


# ---------------------------------------------------------------- Task 2
def rmse(y_true, y_pred):
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sqrt(np.mean(d ** 2))) if d.size else float("nan")


def skor_asimetris(y_true, y_pred):
    """Jumlah (bukan rata-rata) seperti di soal. Lebih kecil lebih baik."""
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def ringkasan_rul(y_true, y_pred):
    """RMSE + skor asimetris total dan per rentang RUL aktual.

    Skor juga dilaporkan sebagai rata-rata per mesin, supaya bisa dibandingkan antar fold
    yang jumlah mesinnya berbeda (skor resmi = jumlah).
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    hasil = {"n": int(len(y_true)), "rmse": rmse(y_true, y_pred),
             "skor": skor_asimetris(y_true, y_pred)}
    hasil["skor_per_mesin"] = hasil["skor"] / max(len(y_true), 1)
    for lo, hi in RENTANG_RUL:
        m = (y_true >= lo) & ((y_true < hi) if hi < RUL_CAP else (y_true <= hi))
        hasil[f"rul_{lo}_{hi}"] = {"n": int(m.sum()), "rmse": rmse(y_true[m], y_pred[m]),
                                   "skor": skor_asimetris(y_true[m], y_pred[m])}
    return hasil


def cakupan_interval(y_true, bawah, atas):
    """Proporsi nilai aktual yang jatuh di dalam interval prediksi."""
    y_true = np.asarray(y_true, float)
    return float(np.mean((y_true >= np.asarray(bawah)) & (y_true <= np.asarray(atas))))


if __name__ == "__main__":
    # uji cepat dengan nilai yang bisa dihitung manual
    assert abs(skor_asimetris([10], [20]) - (np.e - 1)) < 1e-9      # d=+10 -> e^1 - 1
    assert abs(skor_asimetris([20], [7]) - (np.e - 1)) < 1e-9       # d=-13 -> e^1 - 1
    assert skor_asimetris([5], [5]) == 0.0
    assert rmse([0, 0], [3, 4]) == np.sqrt(12.5)
    assert macro_f1([0, 1, 1], [0, 1, 1]) == 1.0
    assert abs(macro_f1([0, 0, 1, 1], [0, 1, 1, 1]) - np.mean([2 / 3, 0.8])) < 1e-9
    print("metrik.py: semua uji lulus")
