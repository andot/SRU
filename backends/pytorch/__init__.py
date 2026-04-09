"""PyTorch backend helpers for loading weights.

This module provides a single helper `load_weights` that mirrors the
previous implementation living in `tools/model/export.py`.
"""
from __future__ import annotations

from collections import OrderedDict
import os
import glob
import torch
from typing import Optional

__all__ = ['load_weights', 'find_checkpoint']


def load_weights(net: torch.nn.Module, ckpt_path: str):
    """Load checkpoint weights into `net` with flexible key handling.

    Tries several common checkpoint layouts (`params_ema`, `params`,
    `state_dict`) and falls back to attempting to strip leading
    `module.` prefixes and finally `strict=False` loading.
    """
    d = torch.load(ckpt_path, map_location='cpu')
    key = None
    if 'params_ema' in d:
        key = 'params_ema'
    elif 'params' in d:
        key = 'params'
    elif 'state_dict' in d:
        key = 'state_dict'
    else:
        key = None

    state = d if key is None else d[key]

    try:
        net.load_state_dict(state, strict=True)
        return
    except Exception as e1:
        print('[Info] direct load failed, attempting key-adapt mapping:', e1)
        new_state = OrderedDict()
        for k, v in state.items():
            nk = k.replace('module.', '')
            new_state[nk] = v

        try:
            net.load_state_dict(new_state, strict=True)
            print('[Info] load_state_dict succeeded after removing "module." prefix')
            return
        except Exception as e2:
            try:
                res = net.load_state_dict(new_state, strict=False)
                missing = getattr(res, 'missing_keys', [])
                unexpected = getattr(res, 'unexpected_keys', [])
                print('[Info] load_state_dict completed with strict=False')
                if missing:
                    print('[Warning] missing keys:', missing[:20])
                if unexpected:
                    print('[Warning] unexpected keys:', unexpected[:20])
                return
            except Exception as e3:
                print('[Error] all load attempts failed:', e3)
                raise


def find_checkpoint(model_name: str, models_root: Optional[str] = None, input_size: Optional[tuple | int] = None, fp16: bool = False) -> Optional[str]:
    """Locate a PyTorch checkpoint (.pth/.pt/.ckpt) under `models_root`.

    Supports `input_size` as:
      - None: no size-specific search
      - int: square size (e.g. 512)
      - (size,) single-element tuple
      - (h, w) two-element tuple

    Also supports `fp16` flag by checking `_fp16` suffixed filenames.
    Returns first matching path or `None`.
    """
    import os

    if models_root is None:
        models_root = os.path.join(os.getcwd(), 'models')

    candidates = []
    base = model_name
    exts = ['.pth', '.pt', '.ckpt', '.pth.tar', '.pt.tar']
    backend_dir = 'pytorch'

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
            # treat equal hw as single size
            size_int = h
        case (int() as h, int() as w):
            size_hw = (h, w)
        case _:
            pass

    # size-specific candidates
    if size_hw is not None:
        h, w = size_hw
        # suffix-based filenames and size directories
        for ext in exts:
            candidates.append(os.path.join(models_root, f"{base}_{h}x{w}{ext}"))
            if fp16:
                candidates.append(os.path.join(models_root, f"{base}_{h}x{w}_fp16{ext}"))
            candidates.append(os.path.join(models_root, f"{h}x{w}", f"{base}{ext}"))
            candidates.append(os.path.join(models_root, backend_dir, f"{h}x{w}", f"{base}{ext}"))
            if fp16:
                candidates.append(os.path.join(models_root, f"{h}x{w}", f"{base}_fp16{ext}"))
                candidates.append(os.path.join(models_root, backend_dir, f"{h}x{w}", f"{base}_fp16{ext}"))

    if size_int is not None:
        s = size_int
        for ext in exts:
            candidates.append(os.path.join(models_root, f"{base}_{s}x{s}{ext}"))
            if fp16:
                candidates.append(os.path.join(models_root, f"{base}_{s}x{s}_fp16{ext}"))
            candidates.append(os.path.join(models_root, str(s), f"{base}{ext}"))
            candidates.append(os.path.join(models_root, backend_dir, str(s), f"{base}{ext}"))
            if fp16:
                candidates.append(os.path.join(models_root, str(s), f"{base}_fp16{ext}"))
                candidates.append(os.path.join(models_root, backend_dir, str(s), f"{base}_fp16{ext}"))

    # generic candidates
    for ext in exts:
        candidates.append(os.path.join(models_root, f"{base}{ext}"))
        if fp16:
            candidates.append(os.path.join(models_root, f"{base}_fp16{ext}"))
        candidates.append(os.path.join(models_root, backend_dir, f"{base}{ext}"))
        if fp16:
            candidates.append(os.path.join(models_root, backend_dir, f"{base}_fp16{ext}"))

    # Glob for filenames containing the model_name
    for pattern in ('*{}*.pth', '*{}*.pt', '*{}*.ckpt', '*{}*.tar'):
        for p in glob.glob(os.path.join(models_root, pattern.format(model_name))):
            candidates.append(p)

    # check candidates
    for p in candidates:
        if os.path.exists(p):
            return p

    # Recursive fallback
    for root, dirs, files in os.walk(models_root):
        for f in files:
            if model_name in f and os.path.splitext(f)[1] in ('.pth', '.pt', '.ckpt', '.tar'):
                return os.path.join(root, f)

    return None
