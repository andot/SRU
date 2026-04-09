"""NCNN backend helpers (find parameter file pairs).

This looks for `.param` files (and corresponding `.bin`) under `models_root`
or `models_root/ncnn` and returns the `.param` path when found.
"""
from __future__ import annotations

import os
import glob
from typing import Optional

__all__ = ['find_checkpoint']


def find_checkpoint(model_name: str, models_root: Optional[str] = None, input_size: Optional[tuple | int] = None, fp16: bool = False) -> Optional[str]:
    if models_root is None:
        models_root = os.path.join(os.getcwd(), 'models')

    search_dirs = [models_root, os.path.join(models_root, 'ncnn')]

    candidates = []
    base = model_name
    backend_dir = 'ncnn'

    size_int = None
    size_hw = None
    match input_size:
        case None:
            pass
        case int() as s:
            size_int = int(s)
        case (int() as s,):
            size_int = int(s)
        case (int() as h, int() as w) if h == w:
            # equal h/w -> single size
            size_int = h
        case (int() as h, int() as w):
            size_hw = (h, w)
        case _:
            pass

    if size_hw is not None:
        h, w = size_hw
        candidates.append(os.path.join(models_root, f"{base}_{h}x{w}.param"))
        candidates.append(os.path.join(models_root, backend_dir, f"{base}_{h}x{w}.param"))
        candidates.append(os.path.join(models_root, f"{h}x{w}", f"{base}.param"))
        candidates.append(os.path.join(models_root, backend_dir, f"{h}x{w}", f"{base}.param"))
        if fp16:
            candidates.append(os.path.join(models_root, f"{base}_{h}x{w}_fp16.param"))
            candidates.append(os.path.join(models_root, backend_dir, f"{base}_{h}x{w}_fp16.param"))

    if size_int is not None:
        s = size_int
        candidates.append(os.path.join(models_root, f"{base}_{s}x{s}.param"))
        candidates.append(os.path.join(models_root, backend_dir, f"{base}_{s}x{s}.param"))
        candidates.append(os.path.join(models_root, str(s), f"{base}.param"))
        candidates.append(os.path.join(models_root, backend_dir, str(s), f"{base}.param"))
        if fp16:
            candidates.append(os.path.join(models_root, f"{base}_{s}x{s}_fp16.param"))
            candidates.append(os.path.join(models_root, backend_dir, str(s), f"{base}_fp16.param"))

    candidates.append(os.path.join(models_root, f"{base}.param"))
    candidates.append(os.path.join(models_root, backend_dir, f"{base}.param"))

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for p in glob.glob(os.path.join(d, f"*{model_name}*.param")):
            candidates.append(p)

    for p in candidates:
        if os.path.exists(p):
            bin_path = os.path.splitext(p)[0] + '.bin'
            if os.path.exists(bin_path):
                return p
    # fallback: accept .param even if .bin missing
    for p in candidates:
        if os.path.exists(p):
            return p

    return None
