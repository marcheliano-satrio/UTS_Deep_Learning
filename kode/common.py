"""Utilitas bersama: seed, device, folder run, simpan config & metrics.

Setiap run menulis ke runs/<nama>/:
- config.json  : argumen + versi library + seed (syarat reproduksibilitas di soal)
- metrics.json : angka yang boleh dikutip di laporan (lihat CLAUDE.md aturan 4)
"""
import json
import os
import platform
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def versi_library():
    v = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ["torch", "torchvision", "numpy", "pandas", "sklearn", "scipy", "umap", "lightgbm"]:
        try:
            v[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            v[m] = None
    try:
        import torch

        v["cuda"] = torch.version.cuda
        v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001
        pass
    return v


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def simpan_json(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def baca_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def buat_run(nama: str, args=None, runs_dir=None):
    """Buat runs/<nama>/ dan tulis config.json. Kembalikan path folder."""
    runs_dir = runs_dir or os.path.join(ROOT, "runs")
    d = os.path.join(runs_dir, nama)
    os.makedirs(d, exist_ok=True)
    cfg = {"nama": nama, "waktu": time.strftime("%Y-%m-%d %H:%M:%S"),
           "argumen": vars(args) if args is not None and hasattr(args, "__dict__") else args,
           "versi": versi_library()}
    simpan_json(cfg, os.path.join(d, "config.json"))
    return d


def _sama(a, b):
    if isinstance(b, bool) or b is None:
        return str(a) == str(b)
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return str(a).replace("\\", "/") == str(b).replace("\\", "/")


def config_beda(run_dir, argumen):
    """Bandingkan daftar argumen CLI yang diminta (mis. ["--fusion", "xattn", "--aug"]) dengan
    runs/<run>/config.json. Kembalikan daftar perbedaan (kosong = sama). Dipakai runner eksperimen
    supaya run lama dengan konfigurasi BERBEDA tidak diam-diam dipakai ulang."""
    p = os.path.join(run_dir, "config.json")
    if not os.path.exists(p):
        return ["config.json tidak ada"]
    lama = baca_json(p).get("argumen") or {}
    beda, i = [], 0
    while i < len(argumen):
        k = argumen[i]
        if not k.startswith("--"):
            i += 1
            continue
        kunci = k[2:].replace("-", "_")
        if i + 1 < len(argumen) and not argumen[i + 1].startswith("--"):
            v, i = argumen[i + 1], i + 2
        else:
            v, i = True, i + 1
        if kunci not in lama or not _sama(v, lama[kunci]):
            beda.append(f"{kunci}: diminta {v}, tersimpan {lama.get(kunci, '<tidak ada>')}")
    return beda


def simpan_metrics(run_dir, metrics: dict):
    simpan_json(metrics, os.path.join(run_dir, "metrics.json"))


def folder_figur(task: int):
    d = os.path.join(ROOT, "laporan_figs", f"task{task}")
    os.makedirs(d, exist_ok=True)
    return d
