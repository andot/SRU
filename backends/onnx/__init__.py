"""ONNX backend helpers."""
from __future__ import annotations

import os
import glob
from typing import Optional

__all__ = ['find_checkpoint']


def find_checkpoint(model_name: str, models_root: Optional[str] = None, input_size: Optional[tuple | int] = None, fp16: bool = False) -> Optional[str]:
    """Locate ONNX models under `models_root` or `models_root/onnx`.

    Returns first matching .onnx path or None.
    """
    if models_root is None:
        models_root = os.path.join(os.getcwd(), 'models')

    search_dirs = [models_root, os.path.join(models_root, 'onnx')]
    candidates = []
    base = model_name
    backend_dir = 'onnx'

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
            # equal h/w treat as single size
            size_int = h
        case (int() as h, int() as w):
            size_hw = (h, w)
        case _:
            pass

    if size_hw is not None:
        h, w = size_hw
        candidates.append(os.path.join(models_root, f"{base}_{h}x{w}.onnx"))
        candidates.append(os.path.join(models_root, backend_dir, f"{base}_{h}x{w}.onnx"))
        candidates.append(os.path.join(models_root, f"{h}x{w}", f"{base}.onnx"))
        candidates.append(os.path.join(models_root, backend_dir, f"{h}x{w}", f"{base}.onnx"))
        if fp16:
            candidates.append(os.path.join(models_root, f"{base}_{h}x{w}_fp16.onnx"))
            candidates.append(os.path.join(models_root, backend_dir, f"{base}_{h}x{w}_fp16.onnx"))

    if size_int is not None:
        s = size_int
        candidates.append(os.path.join(models_root, f"{base}_{s}x{s}.onnx"))
        candidates.append(os.path.join(models_root, backend_dir, f"{base}_{s}x{s}.onnx"))
        candidates.append(os.path.join(models_root, str(s), f"{base}.onnx"))
        candidates.append(os.path.join(models_root, backend_dir, str(s), f"{base}.onnx"))
        if fp16:
            candidates.append(os.path.join(models_root, f"{base}_{s}x{s}_fp16.onnx"))
            candidates.append(os.path.join(models_root, backend_dir, str(s), f"{base}_fp16.onnx"))

    candidates.append(os.path.join(models_root, f"{base}.onnx"))
    candidates.append(os.path.join(models_root, backend_dir, f"{base}.onnx"))

    for d in search_dirs:
        if not os.path.exists(d):
            continue
        for p in glob.glob(os.path.join(d, f"*{model_name}*.onnx")):
            candidates.append(p)

    for p in candidates:
        if os.path.exists(p):
            return p

    return None
