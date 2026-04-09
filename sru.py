#!/usr/bin/env python3
"""
Super Resolution Upscaler
"""

from pathlib import Path
import argparse
import subprocess
import json
import sys
import tempfile
import shutil
import os
import math
from typing import Tuple, List, Optional, Any
import locale
import time
import threading
import queue as pyqueue
import types

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import ncnn as _ncnn

    os.environ["VK_ICD_FILENAMES"] = (
        ""  # ensure no external Vulkan ICDs are loaded by default, we'll handle it in _ncnn_find_real_vulkan_gpu()
    )
except Exception:
    _ncnn = None

_ncnn_vulkan_gpu_cache = None  # cached result: int or None (not yet probed)


def _ncnn_find_real_vulkan_gpu() -> int:
    """Check if ncnn has a usable Vulkan GPU.
    VK_ICD_FILENAMES='' is set at import time to filter out Mock devices.
    Returns GPU index >= 0 on success, or -1 if no usable GPU.
    Suppresses ncnn's stderr output during Vulkan probing.
    Result is cached after first call."""
    global _ncnn_vulkan_gpu_cache
    if _ncnn_vulkan_gpu_cache is not None:
        return _ncnn_vulkan_gpu_cache

    if _ncnn is None:
        _ncnn_vulkan_gpu_cache = -1
        return -1

    # Suppress ncnn stderr output (Vulkan loader messages, device info, etc.)
    _saved = -1
    _devnull = -1
    try:
        _saved = os.dup(2)
        _devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_devnull, 2)
    except Exception:
        pass

    try:
        count = _ncnn.get_gpu_count()
        if count > 0:
            _ncnn_vulkan_gpu_cache = 0
            return 0
    except Exception:
        pass
    finally:
        # Restore stderr
        if _saved >= 0:
            try:
                os.dup2(_saved, 2)
                os.close(_saved)
            except Exception:
                pass
        if _devnull >= 0:
            try:
                os.close(_devnull)
            except Exception:
                pass

    _ncnn_vulkan_gpu_cache = -1
    return -1


try:
    import onnxruntime as _ort
except Exception:
    _ort = None


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
TARGET_4K = (3840, 2160)
COREML_TILE_SIZES = [128, 256, 384, 512]


def _detect_lang() -> str:
    # Prefer using setlocale/getlocale (getdefaultlocale is deprecated)
    try:
        try:
            locale.setlocale(locale.LC_ALL, "")
        except Exception:
            pass
        loc = locale.getlocale()[0]
    except Exception:
        loc = None

    # Fallback to environment variables
    if not loc:
        loc = (
            os.environ.get("LC_ALL")
            or os.environ.get("LANG")
            or os.environ.get("LANGUAGE")
        )

    # On macOS try system defaults as a last resort
    if not loc and sys.platform == "darwin":
        try:
            p = subprocess.run(
                ["defaults", "read", "-g", "AppleLocale"],
                capture_output=True,
                text=True,
                timeout=1,
            )
            if p.returncode == 0:
                loc = p.stdout.strip()
        except Exception:
            pass

    loc = (loc or "").lower()
    if loc.startswith("zh"):
        return "zh"
    if loc.startswith("en"):
        return "en"
    return "other"


LANG = _detect_lang()

MESSAGES = {
    "realesrgan_not_found_msg": {
        "en": "Real-ESRGAN Python package not found or failed to initialize.\nYou can install Real-ESRGAN for GPU acceleration, or allow the script to fall back to slower Pillow-based resizing.",
        "zh": "未检测到 Real-ESRGAN Python 包或初始化失败。\n你可以安装 Real-ESRGAN 以启用 GPU 加速，或者允许脚本回退到较慢的 Pillow 缩放。",
        "other": "Real-ESRGAN not available; will offer Pillow fallback or installation instructions.",
    },
    "fallback_prompt": {
        "en": "Real-ESRGAN not available — fall back to Pillow resize? [Y/n]: ",
        "zh": "未安装 Real-ESRGAN——是否回退为 Pillow 缩放？[Y/n]: ",
        "other": "Real-ESRGAN missing — fall back to Pillow? [Y/n]: ",
    },
    "install_instructions": {
        "en": (
            "\n=== Installation instructions for Real-ESRGAN (summary) ===\n"
            "1) Install PyTorch with ROCm (AMD) or CUDA (NVIDIA) support from https://pytorch.org\n"
            "2) Clone Real-ESRGAN and install requirements:\n   git clone https://github.com/xinntao/Real-ESRGAN.git\n   cd Real-ESRGAN\n   pip install -r requirements.txt\n   pip install -e .\n"
            "3) Download model weights into ./models/ (see releases)\n"
        ),
        "zh": (
            "\n=== Real-ESRGAN 安装指南（摘要） ===\n"
            "1) 根据你的 GPU 安装带有 ROCm（AMD）或 CUDA（NVIDIA）支持的 PyTorch（参考 https://pytorch.org）\n"
            "2) 获取并安装 Real-ESRGAN：\n   git clone https://github.com/xinntao/Real-ESRGAN.git\n   cd Real-ESRGAN\n   pip install -r requirements.txt\n   pip install -e .\n"
            "3) 下载模型权重到 ./models/ 目录（参考 Releases 页面）\n"
        ),
        "other": (
            "See Real-ESRGAN repository for installation instructions: https://github.com/xinntao/Real-ESRGAN\n"
        ),
    },
    "device": {
        "en": "Device: {device}",
        "zh": "设备: {device}",
        "other": "Device: {device}",
    },
    "processing": {
        "en": "Processing: {path}",
        "zh": "处理: {path}",
        "other": "Processing: {path}",
    },
    "dry_run": {
        "en": "Dry-run: target {tw}x{th} suffix {suffix}",
        "zh": "模拟运行: 目标 {tw}x{th} 后缀 {suffix}",
        "other": "Dry-run: target {tw}x{th} suffix {suffix}",
    },
    "input_info": {
        "en": "Input: {w}x{h} @ {fps}fps, frames={nb}",
        "zh": "输入: {w}x{h} @{fps}fps, 帧数={nb}",
        "other": "Input: {w}x{h} @ {fps}fps, frames={nb}",
    },
    "target_info": {
        "en": "Target: {tw}x{th}, suffix={suf}",
        "zh": "目标: {tw}x{th}, 后缀={suf}",
        "other": "Target: {tw}x{th}, suffix={suf}",
    },
    "saved": {"en": "Saved: {path}", "zh": "已保存: {path}", "other": "Saved: {path}"},
    "no_input": {
        "en": "No input files found",
        "zh": "未找到输入文件",
        "other": "No input files found",
    },
}

MESSAGES["no_realesrgan_no_fallback"] = {
    "en": "Real-ESRGAN not available and Pillow fallback not allowed",
    "zh": "未检测到 Real-ESRGAN 且不允许使用 Pillow 回退",
    "other": "Real-ESRGAN not available and Pillow fallback not allowed",
}
MESSAGES["pillow_required"] = {
    "en": "Pillow is required for fallback upscaling",
    "zh": "需要安装 Pillow 用于回退缩放",
    "other": "Pillow is required for fallback upscaling",
}


def t(key: str, **kwargs) -> str:
    template = MESSAGES.get(key, {}).get(LANG) or MESSAGES.get(key, {}).get("en") or ""
    try:
        return template.format(**kwargs)
    except Exception:
        return template


def find_input_files(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
        return [path]
    if path.is_dir():
        files = [p for p in path.rglob("*") if p.suffix.lower() in VIDEO_EXTS]
        return sorted(files)
    return []


def has_rocm() -> bool:
    try:
        return shutil.which("rocm-smi") is not None
    except Exception:
        return False


def detect_best_device() -> str:
    # Improved device detection
    # Priority: mps (Apple) -> rocm (AMD) -> cuda (NVIDIA) -> directml (Windows) -> vulkan -> cpu
    if torch is not None:
        # mps (Apple) check
        try:
            if (
                hasattr(torch, "backends")
                and getattr(torch.backends, "mps", None) is not None
            ):
                if getattr(torch.backends.mps, "is_available", lambda: False)():
                    return "mps"
        except Exception:
            pass

        # Inspect torch build information: prefer ROCm if hip is present
        try:
            tor_version = getattr(torch, "version", None)
            has_hip = getattr(tor_version, "hip", None) is not None
            has_cuda_ver = getattr(tor_version, "cuda", None) is not None
        except Exception:
            has_hip = False
            has_cuda_ver = False

        # If PyTorch build indicates HIP/ROCm, prefer rocm
        if has_hip:
            return "rocm"

        # If CUDA is available and CUDA version exists, prefer cuda
        try:
            cuda_available = hasattr(torch, "cuda") and torch.cuda.is_available()
        except Exception:
            cuda_available = False
        if cuda_available and has_cuda_ver:
            return "cuda"

    # DirectML check: if torch_directml or DmlExecutionProvider is available, prefer directml
    if sys.platform == "win32":
        try:
            import torch_directml

            return "directml"
        except Exception:
            pass
        try:
            if _ort and "DmlExecutionProvider" in _ort.get_available_providers():
                return "directml"
        except Exception:
            pass

    # Check for rocm tools on PATH
    if has_rocm():
        return "rocm"

    # Check for Vulkan via ncnn
    if _ncnn is not None:
        try:
            if _ncnn_find_real_vulkan_gpu() >= 0:
                return "vulkan"
        except Exception:
            pass

    return "cpu"


def probe_video(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr or p.stdout}")
    data = json.loads(p.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No streams found")
    vs = None
    for s in streams:
        if s.get("codec_type") == "video":
            vs = s
            break
    if vs is None:
        raise RuntimeError("No video stream")
    width = int(vs.get("width"))
    height = int(vs.get("height"))
    # frame rate parsing
    r = vs.get("r_frame_rate") or vs.get("avg_frame_rate") or "0/1"
    try:
        num, den = map(int, r.split("/"))
        fps = num / den if den else float(num)
    except Exception:
        fps = 30.0
    # duration: prefer stream duration, then format duration
    duration = None
    try:
        d = vs.get("duration") or data.get("format", {}).get("duration")
        if d is not None:
            duration = float(d)
    except Exception:
        pass
    nb_frames = int(vs.get("nb_frames") or (duration * fps if duration else 0) or 0)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "nb_frames": nb_frames,
        "has_audio": has_audio,
        "duration": duration,
    }


def calculate_target_resolution(w: int, h: int, scale: int = 4) -> Tuple[int, int, str]:
    scaled_w, scaled_h = w * scale, h * scale
    tw, th = TARGET_4K
    if scaled_w <= tw and scaled_h <= th:
        return scaled_w, scaled_h, f"_{scale}x"
    # need to fit into 4K preserving aspect
    # if both exceed, scale to 4K using original aspect
    aspect = w / h
    # try width-limited
    if scaled_w > tw and scaled_h > th:
        # set height = 2160, compute width
        nw = int(tw * w / h)
        return nw, th, "_4k"
    if scaled_w > tw and scaled_h <= th:
        # width exceed -> set width to 3840
        nh = int(tw * h / w)
        return tw, nh, "_4k"
    if scaled_w <= tw and scaled_h > th:
        nw = int(th * w / h)
        return nw, th, "_4k"
    return tw, th, "_4k"


def _available_coreml_sizes(model_name: str) -> List[int]:
    """Scan models/coreml/{size}/ directories for available CoreML model variants."""
    available = []
    for s in COREML_TILE_SIZES:
        d = os.path.join("models", "coreml", str(s))
        for ext in (".mlpackage", ".mlmodel"):
            if os.path.exists(os.path.join(d, model_name + ext)):
                available.append(s)
                break
    return available


def _pick_quality_tile(w: int, h: int, available_sizes: List[int]) -> int:
    """Pick tile size that minimizes total padded pixels. Prefer smaller on tie (less memory)."""
    best = available_sizes[0]
    best_px = float("inf")
    for s in available_sizes:
        pw = math.ceil(w / s) * s
        ph = math.ceil(h / s) * s
        total = pw * ph
        if total < best_px or (total == best_px and s < best):
            best_px = total
            best = s
    return best


def _pick_speed_size(w: int, h: int, available_sizes: List[int]) -> int:
    """Pick model size for speed mode: largest available size not exceeding video's
    max dimension. Falls back to smallest available."""
    video_max = max(w, h)
    candidates = sorted([s for s in available_sizes if s <= video_max], reverse=True)
    if not candidates:
        return min(available_sizes)
    return candidates[0]


def _find_onnx_model(model_name: str) -> Optional[str]:
    """Find ONNX model file in models/onnx/ directory."""
    p = os.path.join("models", "onnx", model_name + ".onnx")
    if os.path.exists(p):
        return os.path.abspath(p)
    return None


def _pick_ort_providers(device: str) -> List[Any]:
    """Pick ONNX Runtime execution providers based on device."""
    available = _ort.get_available_providers() if _ort else []
    if device in ("directml", "cuda", "rocm") and "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "rocm" and "ROCMExecutionProvider" in available:
        return ["ROCMExecutionProvider", "CPUExecutionProvider"]
    if device == "mps" and "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def _find_ncnn_model(model_name: str) -> Optional[Tuple[str, str]]:
    """Find ncnn model files (.param + .bin) in models/ncnn/ directory."""
    base = os.path.join("models", "ncnn")
    param = os.path.join(base, model_name + ".param")
    bin_file = os.path.join(base, model_name + ".bin")
    if os.path.exists(param) and os.path.exists(bin_file):
        return (param, bin_file)
    return None


def _parse_ncnn_param(param_path: str) -> Tuple[str, str]:
    """Parse ncnn .param file to get input/output blob names."""
    input_name = "input"
    output_name = "output"
    try:
        with open(param_path, "r") as f:
            _magic = f.readline()  # magic number
            _counts = f.readline()  # layer_count blob_count
            last_out = None
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                layer_type = parts[0]
                in_count = int(parts[2])
                out_count = int(parts[3])
                offset = 4 + in_count  # skip input blobs
                out_blobs = parts[offset : offset + out_count]
                if layer_type == "Input" and out_blobs:
                    input_name = out_blobs[0]
                if out_blobs:
                    last_out = out_blobs[-1]
            if last_out:
                output_name = last_out
    except Exception:
        pass
    return input_name, output_name


class SimpleProcessor:
    """Wrap realesrgan if available, else Pillow fallback with interactive prompt and instructions."""

    def __init__(
        self,
        model_name: str,
        scale: int,
        device: str,
        half: bool = False,
        tile: int = 0,
        compile_model: bool = False,
        backend: str = "auto",
        coreml_tile_size: int = None,
    ):
        self.model_name = model_name
        self.scale = int(scale)
        self.device = device
        self.backend = backend
        self._use_realesrgan = False
        self.half = bool(half)
        self.tile = int(tile)
        self.compile_model = bool(compile_model)
        # channels_last support removed (no benefit observed on DirectML)

        if backend == "pillow":
            self.model = None
            self._use_coreml = False
            self._use_onnx = False
            self._use_ncnn = False
            if Image is None:
                raise SystemExit("[Error] Pillow not installed")
            return

        try:
            if backend not in ("auto", "pth"):
                raise Exception("skip PyTorch init")
            # Ensure compatibility shim: some torchvision versions lack
            # `torchvision.transforms.functional_tensor`. basicsr may import
            # that module; if it's missing, create a small proxy module that
            # maps required functions to `torchvision.transforms.functional`.
            try:
                import torchvision.transforms.functional_tensor as _ft  # type: ignore
            except Exception:
                try:
                    import torchvision.transforms.functional as _f  # type: ignore

                    mod = types.ModuleType("torchvision.transforms.functional_tensor")

                    # provide minimal function(s) that basicsr expects
                    def rgb_to_grayscale(tensor, num_output_channels=1):
                        return _f.rgb_to_grayscale(
                            tensor, num_output_channels=num_output_channels
                        )

                    mod.rgb_to_grayscale = rgb_to_grayscale
                    import sys as _sys

                    _sys.modules["torchvision.transforms.functional_tensor"] = mod
                except Exception:
                    # couldn't build shim; continue and let the import error surface
                    pass

            # Try common package name
            # Try common package name(s)
            mod = None
            for name in ("realesrgan", "RealESRGAN", "realesrgan_pytorch"):
                try:
                    mod = __import__(name)
                    break
                except Exception:
                    mod = None

            self.model = None
            if mod is not None:
                # Try several common constructor names and call signatures
                candidate_names = ["RealESRGAN", "RealESRGANer", "RealESRGANer"]
                constructor = None
                for cname in candidate_names:
                    # try module level
                    constructor = getattr(mod, cname, None)
                    if constructor is None:
                        # try submodule `realesrgan_model` if present
                        sub = getattr(mod, "realesrgan_model", None)
                        if sub is not None:
                            constructor = getattr(sub, cname, None)
                    if constructor is not None:
                        break

                if constructor is not None:
                    base = os.path.join("models")
                    model_dir = os.path.join(base, model_name)
                    candidates = [
                        model_dir,
                        model_dir + ".pth",
                        model_dir + ".pt",
                        os.path.join(model_dir, "model.pth"),
                        os.path.join(base, model_name + ".pth"),
                        os.path.join(base, model_name + ".pt"),
                    ]

                    # filter to existing files/dirs
                    existing = [p for p in candidates if os.path.exists(p)]
                    tried = []

                    if not existing:
                        print(
                            f"\n[Info] No model weight path found for '{model_name}' under ./models/. Searched candidates:"
                        )
                        for p in candidates:
                            print("  -", p)
                        # also list contents of models dir
                        try:
                            print("\nAvailable files in ./models/:")
                            for p in sorted(os.listdir(base)):
                                print("  -", os.path.join(base, p))
                        except Exception:
                            pass
                    else:
                        # try to build a network model if constructor is RealESRGANer
                        for mp in existing:
                            instantiated = False
                            try:
                                # If constructor is RealESRGANer, it expects a 'model' instance
                                cname = getattr(constructor, "__name__", "")
                                if cname == "RealESRGANer" or "RealESRGANer" in repr(
                                    constructor
                                ):
                                    # build model according to model_name (mirror inference_realesrgan.py)
                                    try:
                                        from basicsr.archs.rrdbnet_arch import RRDBNet
                                    except Exception:
                                        RRDBNet = None
                                    try:
                                        from realesrgan.archs.srvgg_arch import (
                                            SRVGGNetCompact,
                                        )
                                    except Exception:
                                        SRVGGNetCompact = None

                                    net = None
                                    netscale = None
                                    mn = model_name.split(".")[0]
                                    if (
                                        mn == "RealESRGAN_x4plus_anime_6B"
                                        and RRDBNet is not None
                                    ):
                                        net = RRDBNet(
                                            num_in_ch=3,
                                            num_out_ch=3,
                                            num_feat=64,
                                            num_block=6,
                                            num_grow_ch=32,
                                            scale=4,
                                        )
                                        netscale = 4
                                    elif (
                                        mn == "RealESRGAN_x4plus"
                                        and RRDBNet is not None
                                    ):
                                        net = RRDBNet(
                                            num_in_ch=3,
                                            num_out_ch=3,
                                            num_feat=64,
                                            num_block=23,
                                            num_grow_ch=32,
                                            scale=4,
                                        )
                                        netscale = 4
                                    elif (
                                        mn == "RealESRNet_x4plus"
                                        and RRDBNet is not None
                                    ):
                                        net = RRDBNet(
                                            num_in_ch=3,
                                            num_out_ch=3,
                                            num_feat=64,
                                            num_block=23,
                                            num_grow_ch=32,
                                            scale=4,
                                        )
                                        netscale = 4
                                    elif (
                                        mn == "RealESRGAN_x2plus"
                                        and RRDBNet is not None
                                    ):
                                        net = RRDBNet(
                                            num_in_ch=3,
                                            num_out_ch=3,
                                            num_feat=64,
                                            num_block=23,
                                            num_grow_ch=32,
                                            scale=2,
                                        )
                                        netscale = 2
                                    elif (
                                        mn == "realesr-animevideov3"
                                        and SRVGGNetCompact is not None
                                    ):
                                        net = SRVGGNetCompact(
                                            num_in_ch=3,
                                            num_out_ch=3,
                                            num_feat=64,
                                            num_conv=16,
                                            upscale=4,
                                            act_type="prelu",
                                        )
                                        netscale = 4
                                    elif (
                                        mn
                                        in (
                                            "realesr-general-x4v3",
                                            "realesr-general-wdn-x4v3",
                                        )
                                        and SRVGGNetCompact is not None
                                    ):
                                        net = SRVGGNetCompact(
                                            num_in_ch=3,
                                            num_out_ch=3,
                                            num_feat=64,
                                            num_conv=32,
                                            upscale=4,
                                            act_type="prelu",
                                        )
                                        netscale = 4
                                    elif (
                                        mn == "RealESRGANv2-animevideo-xsx4"
                                        and SRVGGNetCompact is not None
                                    ):
                                        net = SRVGGNetCompact(
                                            num_in_ch=3,
                                            num_out_ch=3,
                                            num_feat=64,
                                            num_conv=16,
                                            upscale=4,
                                            act_type="prelu",
                                        )
                                        netscale = 4
                                    elif (
                                        mn == "RealESRGANv2-animevideo-xsx2"
                                        and SRVGGNetCompact is not None
                                    ):
                                        net = SRVGGNetCompact(
                                            num_in_ch=3,
                                            num_out_ch=3,
                                            num_feat=64,
                                            num_conv=16,
                                            upscale=2,
                                            act_type="prelu",
                                        )
                                        netscale = 2

                                    if net is not None:
                                        # determine half precision for device
                                        dev = None
                                        try:
                                            import torch as _torch

                                            if self.device == "mps":
                                                dev = _torch.device("mps")
                                            elif self.device == "cuda":
                                                dev = _torch.device("cuda")
                                            elif self.device == "rocm":
                                                dev = (
                                                    _torch.device("cuda")
                                                    if _torch.cuda.is_available()
                                                    else _torch.device("cpu")
                                                )
                                            elif self.device == "directml":
                                                try:
                                                    import torch_directml

                                                    dev = torch_directml.device()
                                                except ImportError:
                                                    dev = _torch.device("cpu")
                                            else:
                                                dev = _torch.device("cpu")
                                        except Exception:
                                            dev = None
                                        # instantiate RealESRGANer with model
                                        try:
                                            import warnings

                                            with warnings.catch_warnings():
                                                warnings.filterwarnings(
                                                    "ignore",
                                                    message=".*weights_only.*",
                                                    category=FutureWarning,
                                                )
                                                self.model = constructor(
                                                    scale=netscale,
                                                    model_path=mp,
                                                    model=net,
                                                    tile=self.tile,
                                                    tile_pad=10,
                                                    pre_pad=10,
                                                    half=self.half,
                                                    device=dev,
                                                )
                                            self._use_realesrgan = True
                                            self._pth_model_path = mp
                                            self._pth_device = dev
                                            instantiated = True
                                        except Exception as e:
                                            tried.append(
                                                (mp, "RealESRGANer with model", str(e))
                                            )
                                    else:
                                        tried.append(
                                            (
                                                mp,
                                                "no matching net class available",
                                                "skipped",
                                            )
                                        )
                            except Exception as e:
                                tried.append((mp, "exception during build", str(e)))
                            if instantiated:
                                break

                    if not self._use_realesrgan and existing:
                        print(
                            "\n[Debug] Found realesrgan module but failed to instantiate model.\n"
                            "Module: %s file=%s"
                            % (
                                getattr(mod, "__name__", str(mod)),
                                getattr(mod, "__file__", None),
                            )
                        )
                        print(
                            "Tried candidate model files and constructor args with resulting errors:"
                        )
                        for mp, args, err in tried:
                            print(" ", mp, args, "->", err)
                    # if we did instantiate, try to detect preferred call method
                    if self._use_realesrgan and self.model is not None:
                        # prefer predict, else enhance, else call
                        if hasattr(self.model, "predict"):
                            self._realesrgan_call = "predict"
                        elif hasattr(self.model, "enhance"):
                            self._realesrgan_call = "enhance"
                        elif callable(self.model):
                            self._realesrgan_call = "__call__"
                        else:
                            self._realesrgan_call = None
                        # Determine DirectML/device hints early for downstream decisions
                        pth_dev = getattr(self, "_pth_device", None)
                        device_str = str(getattr(self, "device", "")).lower()
                        detect_dml = (
                            ("directml" in device_str)
                            or (
                                pth_dev is not None
                                and "directml" in str(pth_dev).lower()
                            )
                            or ("torch_directml" in sys.modules)
                        )

                        # channels_last conversion removed (no runtime benefit observed)

                        # Optionally try to compile the underlying network (PyTorch 2.x)
                        # Skip torch.compile when running on DirectML because it can
                        # interact badly with opaque/driver-specific tensors.
                        if self.compile_model and self._realesrgan_call is not None:
                            try:
                                if detect_dml:
                                    print(
                                        "[Info] Skipping torch.compile on DirectML device (unsupported)"
                                    )
                                else:
                                    if torch is not None and hasattr(torch, "compile"):
                                        net = getattr(self.model, "model", None)
                                        if net is not None:
                                            print(
                                                "[Info] Attempting torch.compile on network (may improve performance)"
                                            )
                                            try:
                                                compiled = torch.compile(net)
                                                setattr(self.model, "model", compiled)
                                                print("[Info] torch.compile succeeded")
                                            except Exception as e:
                                                print(
                                                    "[Warning] torch.compile failed:", e
                                                )
                                    else:
                                        print(
                                            "[Info] torch.compile not available in this torch build"
                                        )
                            except Exception:
                                pass

                        # If running on DirectML, replace all PReLU modules with a
                        # DirectML-friendly implementation using pos + weight * neg
                        try:
                            if detect_dml:
                                try:
                                    import torch as _torch

                                    net = getattr(self.model, "model", self.model)

                                    def _set_child_by_name(root, full_name, new_mod):
                                        parts = (
                                            full_name.split(".") if full_name else []
                                        )
                                        parent = root
                                        if parts:
                                            for p in parts[:-1]:
                                                if p.isdigit():
                                                    parent = parent[int(p)]
                                                else:
                                                    parent = getattr(parent, p)
                                            last = parts[-1]
                                        else:
                                            last = ""
                                        try:
                                            if last.isdigit():
                                                parent[int(last)] = new_mod
                                            elif last:
                                                setattr(parent, last, new_mod)
                                            else:
                                                return False
                                            return True
                                        except Exception:
                                            for n, ch in parent.named_children():
                                                if ch is new_mod:
                                                    try:
                                                        setattr(parent, n, new_mod)
                                                        return True
                                                    except Exception:
                                                        pass
                                            return False

                                    class _PReLU_DML(_torch.nn.Module):
                                        def __init__(self, orig):
                                            super().__init__()
                                            # 获取原始权重并转为 Buffer（不参与梯度计算）
                                            w = orig.weight.detach().clone()
                                            self.register_buffer("_weight", w)
                                            self.w_b = None  # 预存广播后的权重

                                        def forward(self, x):
                                            # 直接传 0 给 maximum/minimum，避免创建 zeros_like 这样的大张量
                                            # DirectML 通常支持 Scalar 广播，这比 Tensor 广播快得多
                                            zeros = x.new_tensor(0.0)
                                            pos = _torch.max(zeros, x)
                                            neg = _torch.min(zeros, x)

                                            # 保持原有的 w_b 逻辑，但在 forward 内直接 view
                                            # 这样 DML 编译器能一眼看出这是在做通道维度的广播
                                            if self._weight.ndim == 1 and x.dim() >= 2:
                                                w_b = self._weight.view(
                                                    [1, -1] + [1] * (x.dim() - 2)
                                                )
                                            else:
                                                w_b = self._weight

                                            return pos + w_b * neg

                                    any_wrapped = False
                                    for name, mod in list(net.named_modules()):
                                        if isinstance(mod, _torch.nn.PReLU):
                                            try:
                                                _set_child_by_name(
                                                    net, name, _PReLU_DML(mod)
                                                )
                                                any_wrapped = True
                                            except Exception:
                                                pass
                                    if any_wrapped:
                                        print(
                                            "[Info] Replaced PReLU modules with DirectML-friendly implementation"
                                        )
                                except Exception as _e:
                                    print("[Debug] PReLU replace error:", _e)
                        except Exception:
                            pass
                else:
                    info = {
                        "module_name": getattr(mod, "__name__", str(mod)),
                        "module_file": getattr(mod, "__file__", None),
                        "dir": [n for n in dir(mod) if not n.startswith("_")][:200],
                    }
                    print(
                        "\n[Debug] realesrgan module detected but no known constructor symbol. Module info:"
                    )
                    print(f"  name: {info['module_name']}")
                    print(f"  file: {info['module_file']}")
                    print(f"  members (sample): {info['dir']}")
        except Exception:
            self.model = None

        if backend == "pth" and not self._use_realesrgan:
            print(t("realesrgan_not_found_msg"))
            self._print_install_instructions()
            raise SystemExit(1)

        # CoreML auto-detect on macOS: prefer CoreML if a matching .mlpackage/.mlmodel exists
        self._use_coreml = False
        self._coreml_model = None
        self._coreml_input_name = None
        self._coreml_output_name = None
        self._coreml_input_size = None
        self._coreml_output_size = None
        try:
            if backend in ("auto", "coreml") and sys.platform == "darwin":
                if coreml_tile_size is not None:
                    coreml_candidates = [
                        os.path.join(
                            "models",
                            "coreml",
                            str(coreml_tile_size),
                            model_name + ".mlpackage",
                        ),
                        os.path.join(
                            "models",
                            "coreml",
                            str(coreml_tile_size),
                            model_name + ".mlmodel",
                        ),
                        os.path.join("models", "coreml", model_name + ".mlpackage"),
                        os.path.join("models", "coreml", model_name + ".mlmodel"),
                    ]
                else:
                    coreml_candidates = [
                        os.path.join("models", "coreml", model_name + ".mlpackage"),
                        os.path.join("models", "coreml", model_name + ".mlmodel"),
                    ]
                for cp in coreml_candidates:
                    if os.path.exists(cp):
                        try:
                            import coremltools as ct

                            try:
                                ml = ct.models.MLModel(
                                    cp, compute_units=ct.ComputeUnit.ALL
                                )
                            except Exception:
                                ml = ct.models.MLModel(cp)
                            spec = ml.get_spec()
                            # pick first input name
                            if spec.description.input:
                                in_desc = spec.description.input[0]
                                in_name = in_desc.name
                                self._coreml_input_name = in_name
                                # try to read shape
                                try:
                                    in_type = in_desc.type
                                    if (
                                        in_type.HasField("multiArrayType")
                                        and in_type.multiArrayType.shape
                                    ):
                                        shape = tuple(
                                            int(x) for x in in_type.multiArrayType.shape
                                        )
                                        self._coreml_input_size = shape
                                except Exception:
                                    self._coreml_input_size = None
                            # pick first output name and shape
                            if spec.description.output:
                                out_desc = spec.description.output[0]
                                self._coreml_output_name = out_desc.name
                                try:
                                    out_type = out_desc.type
                                    if (
                                        out_type.HasField("multiArrayType")
                                        and out_type.multiArrayType.shape
                                    ):
                                        self._coreml_output_size = tuple(
                                            int(x)
                                            for x in out_type.multiArrayType.shape
                                        )
                                except Exception:
                                    pass
                            self._use_coreml = True
                            self._coreml_model = ml
                            print("[Info] Using CoreML model for inference:", cp)
                            break
                        except Exception as e:
                            # CoreML not usable in this env; skip
                            print("[Debug] CoreML load failed for", cp, ":", e)
                            continue
        except Exception:
            self._use_coreml = False

        if backend == "coreml" and not self._use_coreml:
            if sys.platform != "darwin":
                print("[Error] CoreML backend is only available on macOS")
            else:
                print("[Error] CoreML backend requested but no model found")
            raise SystemExit(1)

        # --- Auto backend selection priority ---
        # Round 1 (GPU-accelerated):
        #   1. CoreML (macOS native GPU)
        #   2. ONNX + GPU EP (DML/CUDA/ROCM — not CoreML EP)
        #   3. pth + GPU (DirectML/CUDA/MPS)
        #   4. ncnn + Vulkan GPU
        # Round 2 (CPU / slow GPU fallback):
        #   5. ONNX (CoreML EP / CPU)
        #   6. pth CPU
        #   7. ncnn CPU
        #   8. Pillow

        # --- ONNX Runtime ---
        self._use_onnx = False
        self._onnx_session = None
        self._onnx_input_name = "input"
        self._onnx_output_name = "output"
        self._onnx_cpu_fallback = None
        try:
            if (
                backend in ("auto", "onnx")
                and _ort is not None
                and (backend == "onnx" or not self._use_coreml)
            ):
                onnx_path = _find_onnx_model(model_name)
                if onnx_path:
                    providers = _pick_ort_providers(device)
                    # GPU EPs that are truly fast (exclude CoreMLExecutionProvider)
                    _gpu_eps = {
                        "CUDAExecutionProvider",
                        "ROCMExecutionProvider",
                        "DmlExecutionProvider",
                    }
                    _onnx_has_fast_gpu = providers[0] in _gpu_eps
                    if backend == "onnx" or _onnx_has_fast_gpu:
                        sess = _ort.InferenceSession(onnx_path, providers=providers)
                        self._onnx_session = sess
                        self._onnx_input_name = sess.get_inputs()[0].name
                        self._onnx_output_name = sess.get_outputs()[0].name
                        self._use_onnx = True
                        active_ep = (
                            sess.get_providers()[0]
                            if sess.get_providers()
                            else "unknown"
                        )
                        print(
                            f"[Info] Using ONNX Runtime for inference ({active_ep}): {onnx_path}"
                        )
                    else:
                        # Defer ONNX CPU/CoreML EP to round 2
                        self._onnx_cpu_fallback = (onnx_path, providers)
        except Exception as e:
            print(f"[Debug] ONNX Runtime load failed: {e}")
            self._use_onnx = False

        if backend == "onnx" and not self._use_onnx:
            if _ort is None:
                print(
                    "[Error] onnxruntime not installed (pip install onnxruntime / onnxruntime-gpu)"
                )
            else:
                print(
                    "[Error] ONNX backend requested but no model found in models/onnx/"
                )
            raise SystemExit(1)

        # --- pth GPU check (round 1 step 3) ---
        # In auto mode, check if pth has a real GPU device
        _pth_has_gpu = False
        if self._use_realesrgan and not self._use_coreml and not self._use_onnx:
            pth_dev = getattr(self, "_pth_device", None)
            if pth_dev is not None and str(pth_dev) != "cpu":
                _pth_has_gpu = True
                print(
                    f'[Info] Using PyTorch (pth) for inference ({pth_dev}): {getattr(self, "_pth_model_path", None)}'
                )

        # --- ncnn ---
        self._use_ncnn = False
        self._ncnn_net = None
        self._ncnn_input_name = "input"
        self._ncnn_output_name = "output"
        self._ncnn_cpu_fallback = None
        try:
            _skip_ncnn_auto = backend == "auto" and _pth_has_gpu
            if (
                backend in ("auto", "ncnn")
                and _ncnn is not None
                and (
                    backend == "ncnn"
                    or (
                        not self._use_coreml and not self._use_onnx and not _pth_has_gpu
                    )
                )
                and not _skip_ncnn_auto
            ):
                ncnn_files = _find_ncnn_model(model_name)
                if ncnn_files:
                    if self.device == "cpu":
                        gpu_idx = -1
                    else:
                        gpu_idx = _ncnn_find_real_vulkan_gpu()
                    use_vulkan = gpu_idx >= 0
                    if backend == "ncnn" or use_vulkan:
                        param_path, bin_path = ncnn_files
                        in_name, out_name = _parse_ncnn_param(param_path)
                        net = _ncnn.Net()
                        net.opt.use_vulkan_compute = use_vulkan
                        if use_vulkan and gpu_idx > 0:
                            try:
                                net.set_vulkan_device(gpu_idx)
                            except (AttributeError, Exception):
                                pass
                        net.load_param(param_path)
                        net.load_model(bin_path)
                        self._ncnn_net = net
                        self._ncnn_input_name = in_name
                        self._ncnn_output_name = out_name
                        self._use_ncnn = True
                        vk_str = f" (Vulkan GPU {gpu_idx})" if use_vulkan else " (CPU)"
                        print(
                            f"[Info] Using ncnn model for inference{vk_str}: {param_path}"
                        )
                    else:
                        # Defer ncnn CPU to round 2
                        self._ncnn_cpu_fallback = ncnn_files
        except Exception as e:
            print(f"[Debug] ncnn load failed: {e}")
            self._use_ncnn = False

        if backend == "ncnn" and not self._use_ncnn:
            if _ncnn is None:
                print("[Error] ncnn not installed (pip install ncnn)")
            else:
                print(
                    "[Error] ncnn backend requested but no model found in models/ncnn/"
                )
            raise SystemExit(1)

        # --- Round 2: CPU / slow fallbacks ---
        if (
            not self._use_coreml
            and not self._use_onnx
            and not _pth_has_gpu
            and not self._use_ncnn
        ):
            # 5. ONNX CPU/CoreML EP fallback
            fb = self._onnx_cpu_fallback
            if fb and not self._use_onnx:
                try:
                    onnx_path, providers = fb
                    providers = [
                        (
                            "CoreMLExecutionProvider",
                            {
                                "ModelFormat": "MLProgram",
                                "MLComputeUnits": "ALL",
                                "RequireStaticInputShapes": "0",
                                "EnableOnSubgraphs": "0",
                            },
                        ),
                    ]
                    sess = _ort.InferenceSession(onnx_path, providers=providers)
                    self._onnx_session = sess
                    self._onnx_input_name = sess.get_inputs()[0].name
                    self._onnx_output_name = sess.get_outputs()[0].name
                    self._use_onnx = True
                    active_ep = (
                        sess.get_providers()[0] if sess.get_providers() else "unknown"
                    )
                    print(
                        f"[Info] Using ONNX Runtime for inference ({active_ep}): {onnx_path}"
                    )
                except Exception as e:
                    print(f"[Debug] ONNX Runtime CPU fallback failed: {e}")

            # 6. pth CPU
            if (
                not self._use_onnx
                and self._use_realesrgan
                and not self._use_coreml
                and not self._use_ncnn
            ):
                pth_dev = getattr(self, "_pth_device", None)
                pth_path = getattr(self, "_pth_model_path", None)
                print(
                    f"[Info] Using PyTorch (pth) for inference ({pth_dev}): {pth_path}"
                )

            # 7. ncnn CPU fallback
            if (
                not self._use_onnx
                and not self._use_realesrgan
                and self._ncnn_cpu_fallback
                and not self._use_ncnn
            ):
                try:
                    param_path, bin_path = self._ncnn_cpu_fallback
                    in_name, out_name = _parse_ncnn_param(param_path)
                    net = _ncnn.Net()
                    net.opt.use_vulkan_compute = False
                    net.load_param(param_path)
                    net.load_model(bin_path)
                    self._ncnn_net = net
                    self._ncnn_input_name = in_name
                    self._ncnn_output_name = out_name
                    self._use_ncnn = True
                    print(f"[Info] Using ncnn model for inference (CPU): {param_path}")
                except Exception as e:
                    print(f"[Debug] ncnn CPU fallback failed: {e}")

    @property
    def native_output_size(self):
        """Return (width, height) of model's native output, or None."""
        s = getattr(self, "_coreml_output_size", None)
        if self._use_coreml and s:
            if len(s) == 4:
                return (int(s[3]), int(s[2]))
            elif len(s) == 3:
                return (int(s[2]), int(s[1]))
        return None

    @property
    def native_input_size(self):
        """Return (width, height) of model's expected input, or None."""
        s = getattr(self, "_coreml_input_size", None)
        if self._use_coreml and s:
            if len(s) == 4:
                return (int(s[3]), int(s[2]))
            elif len(s) == 3:
                return (int(s[2]), int(s[1]))
        return None

    def _print_install_instructions(self):
        print(t("install_instructions"))

    def _coreml_infer_one(self, tile):
        """Run CoreML inference on a single tile (HWC uint8) -> HWC float array."""
        inp_name = self._coreml_input_name or "x"
        arr_t = np.transpose(tile, (2, 0, 1)).astype(np.float32)
        arr_t *= 1.0 / 255.0
        arr_t = arr_t[np.newaxis]
        res = self._coreml_model.predict({inp_name: arr_t})
        out = next(iter(res.values())) if isinstance(res, dict) else res
        out_arr = np.array(out)
        if out_arr.ndim == 4:
            out_arr = out_arr[0]
        if out_arr.ndim == 3 and out_arr.shape[0] == 3:
            out_arr = np.transpose(out_arr, (1, 2, 0))
        return out_arr

    def _coreml_tile_process(self, frame, tile_size: int = 0):
        """Process a frame using tiling for CoreML models.
        tile_size: explicit tile size (for dynamic models via --tile); 0 = use model's fixed input shape.
        """
        h, w = frame.shape[:2]
        if tile_size > 0:
            tile_h, tile_w = tile_size, tile_size
        else:
            shape = self._coreml_input_size
            if len(shape) == 4:
                tile_h, tile_w = int(shape[2]), int(shape[3])
            else:
                tile_h, tile_w = int(shape[1]), int(shape[2])
        scale = self.scale
        # Pad to multiples of tile size using reflection
        pad_h = (tile_h - h % tile_h) % tile_h
        pad_w = (tile_w - w % tile_w) % tile_w
        if pad_h or pad_w:
            padded = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            padded = frame
        ph, pw = padded.shape[:2]
        tiles_y = ph // tile_h
        tiles_x = pw // tile_w
        # Process tiles row by row and concatenate
        out_tile_h = tile_h * scale
        out_tile_w = tile_w * scale
        rows = []
        for yi in range(tiles_y):
            cols = []
            for xi in range(tiles_x):
                tile = padded[
                    yi * tile_h : (yi + 1) * tile_h, xi * tile_w : (xi + 1) * tile_w, :
                ]
                out_tile = self._coreml_infer_one(tile)
                cols.append(out_tile[:out_tile_h, :out_tile_w, :])
            rows.append(np.concatenate(cols, axis=1))
        output = np.concatenate(rows, axis=0)
        # Crop to actual output size
        output = output[: h * scale, : w * scale, :]
        # Convert to uint8
        if output.dtype != np.uint8:
            output = np.clip(output, 0.0, 1.0)
            output *= 255.0
            output = (output + 0.5).astype(np.uint8)
        return output

    def _ncnn_infer_one(self, tile):
        """Run ncnn inference on a single tile (HWC uint8) -> HWC float array."""
        arr = tile.astype(np.float32) / 255.0
        arr = np.ascontiguousarray(
            np.transpose(arr, (2, 0, 1))
        )  # HWC -> CHW, must be contiguous for ncnn.Mat
        mat_in = _ncnn.Mat(arr)
        ex = self._ncnn_net.create_extractor()
        ex.input(self._ncnn_input_name, mat_in)
        ret, mat_out = ex.extract(self._ncnn_output_name)
        out = np.array(mat_out)  # CHW float
        if out.ndim == 3 and out.shape[0] == 3:
            out = np.transpose(out, (1, 2, 0))  # CHW -> HWC
        return out

    def _ncnn_tile_process(self, frame):
        """Process a frame using tiling for ncnn models."""
        h, w = frame.shape[:2]
        tile_size = self.tile
        scale = self.scale
        pad_h = (tile_size - h % tile_size) % tile_size
        pad_w = (tile_size - w % tile_size) % tile_size
        if pad_h or pad_w:
            padded = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            padded = frame
        ph, pw = padded.shape[:2]
        tiles_y = ph // tile_size
        tiles_x = pw // tile_size
        out_tile_size = tile_size * scale
        rows = []
        for yi in range(tiles_y):
            cols = []
            for xi in range(tiles_x):
                tile = padded[
                    yi * tile_size : (yi + 1) * tile_size,
                    xi * tile_size : (xi + 1) * tile_size,
                    :,
                ]
                out_tile = self._ncnn_infer_one(tile)
                cols.append(out_tile[:out_tile_size, :out_tile_size, :])
            rows.append(np.concatenate(cols, axis=1))
        output = np.concatenate(rows, axis=0)
        output = output[: h * scale, : w * scale, :]
        if output.dtype != np.uint8:
            output = np.clip(output, 0.0, 1.0)
            output *= 255.0
            output = (output + 0.5).astype(np.uint8)
        return output

    def _onnx_infer_one(self, tile):
        """Run ONNX Runtime inference on a single tile (HWC uint8) -> HWC float array."""
        arr = tile.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
        arr = arr[np.newaxis]  # add batch dim: [1, 3, H, W]
        out = self._onnx_session.run(
            [self._onnx_output_name], {self._onnx_input_name: arr}
        )[0]
        # out shape: [1, 3, H*scale, W*scale]
        out = out[0]  # remove batch
        if out.ndim == 3 and out.shape[0] == 3:
            out = np.transpose(out, (1, 2, 0))  # CHW -> HWC
        return out

    def _onnx_tile_process(self, frame):
        """Process a frame using tiling for ONNX Runtime models."""
        h, w = frame.shape[:2]
        tile_size = self.tile
        scale = self.scale
        pad_h = (tile_size - h % tile_size) % tile_size
        pad_w = (tile_size - w % tile_size) % tile_size
        if pad_h or pad_w:
            padded = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        else:
            padded = frame
        ph, pw = padded.shape[:2]
        tiles_y = ph // tile_size
        tiles_x = pw // tile_size
        out_tile_size = tile_size * scale
        rows = []
        for yi in range(tiles_y):
            cols = []
            for xi in range(tiles_x):
                tile = padded[
                    yi * tile_size : (yi + 1) * tile_size,
                    xi * tile_size : (xi + 1) * tile_size,
                    :,
                ]
                out_tile = self._onnx_infer_one(tile)
                cols.append(out_tile[:out_tile_size, :out_tile_size, :])
            rows.append(np.concatenate(cols, axis=1))
        output = np.concatenate(rows, axis=0)
        output = output[: h * scale, : w * scale, :]
        if output.dtype != np.uint8:
            output = np.clip(output, 0.0, 1.0)
            output *= 255.0
            output = (output + 0.5).astype(np.uint8)
        return output

    def upscale_batch(self, frames: List, start_idx: int = 0):
        outs = []
        for f in frames:
            # CoreML path (macOS)
            if self._use_coreml and self._coreml_model is not None:
                try:
                    arr = f  # HWC uint8
                    # Check if frame matches model input size
                    shape = getattr(self, "_coreml_input_size", None)
                    need_tile = False
                    dynamic_tile = 0
                    if shape and len(shape) >= 3:
                        th_m = int(shape[2] if len(shape) == 4 else shape[1])
                        tw_m = int(shape[3] if len(shape) == 4 else shape[2])
                        if (th_m, tw_m) != (arr.shape[0], arr.shape[1]):
                            need_tile = True
                    elif not shape and self.tile > 0:
                        # Dynamic CoreML model with --tile specified
                        need_tile = True
                        dynamic_tile = self.tile
                    if need_tile:
                        out_uint8 = self._coreml_tile_process(
                            arr, tile_size=dynamic_tile
                        )
                    else:
                        out_arr = self._coreml_infer_one(arr)
                        if out_arr.dtype != np.uint8:
                            out_arr = np.clip(out_arr, 0.0, 1.0)
                            out_arr *= 255.0
                            out_uint8 = (out_arr + 0.5).astype(np.uint8)
                        else:
                            out_uint8 = out_arr
                    outs.append(out_uint8)
                    continue
                except Exception:
                    # if coreml fails, fall back to other paths
                    pass

            # ONNX Runtime path (cuda/rocm/cpu — faster than PyTorch)
            if self._use_onnx and self._onnx_session is not None:
                try:
                    arr = f  # HWC uint8
                    if self.tile > 0:
                        out_uint8 = self._onnx_tile_process(arr)
                    else:
                        out_arr = self._onnx_infer_one(arr)
                        if out_arr.dtype != np.uint8:
                            out_arr = np.clip(out_arr, 0.0, 1.0)
                            out_arr *= 255.0
                            out_uint8 = (out_arr + 0.5).astype(np.uint8)
                        else:
                            out_uint8 = out_arr
                    outs.append(out_uint8)
                    continue
                except Exception as e:
                    print(f"[Debug] ONNX inference failed: {e}")
                    # if onnx fails, fall back to other paths
                    pass

            # ncnn path (cross-platform Vulkan GPU)
            if self._use_ncnn and self._ncnn_net is not None:
                try:
                    arr = f  # HWC uint8
                    if self.tile > 0:
                        out_uint8 = self._ncnn_tile_process(arr)
                    else:
                        out_arr = self._ncnn_infer_one(arr)
                        if out_arr.dtype != np.uint8:
                            out_arr = np.clip(out_arr, 0.0, 1.0)
                            out_arr *= 255.0
                            out_uint8 = (out_arr + 0.5).astype(np.uint8)
                        else:
                            out_uint8 = out_arr
                    outs.append(out_uint8)
                    continue
                except Exception as e:
                    print(f"[Debug] NCNN inference failed: {e}")
                    # if ncnn fails, fall back to other paths
                    pass

            if self._use_realesrgan and self.model is not None:
                try:
                    # Suppress Real-ESRGAN's internal "Tile X/Y" prints
                    import io as _io

                    _orig_stdout = sys.stdout
                    sys.stdout = _io.StringIO()
                    try:
                        # call the detected interface
                        if getattr(self, "_realesrgan_call", None) == "predict":
                            out = self.model.predict(f)
                        elif getattr(self, "_realesrgan_call", None) == "enhance":
                            out = self.model.enhance(f)
                        elif getattr(self, "_realesrgan_call", None) == "__call__":
                            out = self.model(f)
                        else:
                            # fallback: try common method names
                            if hasattr(self.model, "predict"):
                                out = self.model.predict(f)
                            elif hasattr(self.model, "enhance"):
                                out = self.model.enhance(f)
                            elif callable(self.model):
                                out = self.model(f)
                            else:
                                raise RuntimeError(
                                    "Realesrgan model has no callable predict/enhance"
                                )
                    finally:
                        sys.stdout = _orig_stdout
                    # normalize output
                    if isinstance(out, tuple) or isinstance(out, list):
                        out = out[0]
                    outs.append(np.array(out).astype("uint8"))
                    continue
                except Exception as e:
                    # If DirectML is in use, run a diagnostic trace to stderr to help locate failing op
                    try:
                        sys.stderr.write(
                            f"[Debug] Exception in realesrgan (pth): {e}\n"
                        )
                        # detect if this processor was using torch-directml
                        ptd = getattr(self, "_pth_device", None)
                        is_dml = False
                        try:
                            if ptd is not None and "directml" in str(ptd).lower():
                                is_dml = True
                        except Exception:
                            pass
                        try:
                            if str(getattr(self, "device", "")).lower() == "directml":
                                is_dml = True
                        except Exception:
                            pass
                        if is_dml:
                            try:
                                _diag_torch_directml_model(
                                    self.model,
                                    f,
                                    model_name=getattr(self, "model_name", None),
                                )
                            except Exception as de:
                                sys.stderr.write(
                                    f"[Debug] DirectML diagnostic failed: {de}\n"
                                )
                    except Exception:
                        pass
                    # fall through to Pillow if allowed
                    pass
            if self.backend not in ("auto", "pillow"):
                raise RuntimeError(f'Backend "{self.backend}" failed for this frame')
            # Pillow fallback
            if Image is None:
                raise RuntimeError(t("pillow_required"))
            img = Image.fromarray(f)
            nw = int(img.width * self.scale)
            nh = int(img.height * self.scale)
            img = img.resize((nw, nh), Image.LANCZOS)
            outs.append(np.array(img))
        return outs


def run_one(
    input_path: Path,
    output_path: Optional[Path],
    device: str,
    model_choice: Optional[str],
    batch_size: int,
    dry_run: bool,
    tile: Optional[int] = None,
    half: bool = False,
    compile_model: bool = False,
    backend: str = "auto",
    _processor_cache=None,
    shrink_size: int = 0,
):
    info = probe_video(input_path)
    w, h = info["width"], info["height"]
    fps = info["fps"] or 30.0
    nb_frames = info["nb_frames"]
    has_audio = info["has_audio"]
    duration = info.get("duration")

    # choose model
    if model_choice and model_choice != "auto":
        model = model_choice
    else:
        model = "realesr-animevideov3"

    # detect model upscale factor from name
    mn = model.split(".")[0]
    model_scale = 2 if "x2" in mn.lower() else 4

    tw, th, suffix = calculate_target_resolution(w, h, model_scale)
    print(t("input_info", w=w, h=h, fps=fps, nb=nb_frames))
    print(t("target_info", tw=tw, th=th, suf=suffix))

    if dry_run:
        return str(input_path), (tw, th), suffix

    # Determine speed mode shrink target (applies to ALL backends, not just CoreML)
    # shrink_size: 0 = off, -1 = auto, >0 = explicit shrink size
    speed_target = 0
    if shrink_size != 0:
        if shrink_size > 0:
            if shrink_size >= max(w, h):
                print(
                    f"[Info] Speed mode: shrink size {shrink_size} >= video max dimension {max(w, h)}, skipping shrink"
                )
                speed_target = 0
            else:
                speed_target = shrink_size
        else:
            # auto: pick best shrink size
            speed_target = _pick_speed_size(w, h, COREML_TILE_SIZES)
        if speed_target > 0:
            print(f"[Info] Speed mode target: {speed_target}x{speed_target}")

    # Determine optimal CoreML tile size (CoreML is macOS only)
    if backend not in ("auto", "coreml") or (
        backend == "auto" and sys.platform != "darwin"
    ):
        coreml_ts = None
    else:
        available_sizes = _available_coreml_sizes(model)
        if available_sizes:
            # If user explicitly specified --tile and it matches a fixed model, use it;
            # if --tile is specified but not in fixed sizes, fall back to dynamic CoreML.
            if tile is not None and tile > 0 and tile in available_sizes:
                coreml_ts = tile
            elif tile is not None and tile > 0:
                # tile size not in fixed sizes -> use dynamic CoreML model
                coreml_ts = None
            elif tile == 0:
                # explicit -t 0: no tiling, use dynamic model
                coreml_ts = None
            elif shrink_size != 0:
                # In speed mode, try to match speed_target to a fixed model
                if speed_target > 0 and speed_target in available_sizes:
                    coreml_ts = speed_target
                else:
                    coreml_ts = None  # dynamic
            else:
                coreml_ts = _pick_quality_tile(w, h, available_sizes)
            if coreml_ts is not None:
                print(f"[Info] CoreML tile size: {coreml_ts}x{coreml_ts}")
            else:
                effective_tile = tile if tile is not None and tile > 0 else speed_target
                print(f"[Info] Using dynamic CoreML model (tile={effective_tile})")
        else:
            coreml_ts = None

    cache_key = (model, coreml_ts, backend)
    if _processor_cache is not None and cache_key in _processor_cache:
        processor = _processor_cache[cache_key]
    else:
        print("[Info] Loading model...")
        processor = SimpleProcessor(
            model,
            scale=model_scale,
            device=device,
            half=half,
            tile=(tile or 0),
            compile_model=compile_model,
            backend=backend,
            coreml_tile_size=coreml_ts,
        )
        if _processor_cache is not None:
            _processor_cache[cache_key] = processor

    # prepare temporary output video (video only)
    tmp_vid = Path(tempfile.mkstemp(suffix=".mp4")[1])

    # Detect best available encoder: prefer hardware encoders, fallback to libx264 ultrafast
    def _probe_encoder(enc_args):
        """Test if an encoder actually works by encoding a tiny synthetic clip."""
        try:
            cmd = [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=256x256:d=0.04:r=25",
                "-pix_fmt",
                "yuv420p",
                *enc_args,
                "-f",
                "null",
                "-",
            ]
            p = subprocess.run(cmd, capture_output=True, timeout=1)
            return p.returncode == 0
        except Exception as e:
            print(f"[Debug] Encoder probe failed for {' '.join(enc_args)}: {e}")
            return False

    def _pick_encoder():
        # Try all HW encoders in order; _probe_encoder ensures they actually work
        candidates = []
        # macOS: VideoToolbox
        if sys.platform == "darwin":
            candidates.append(["-c:v", "hevc_videotoolbox", "-q:v", "65"])
            candidates.append(["-c:v", "h264_videotoolbox", "-q:v", "65"])
        # All platforms: try NVENC, AMF, QSV
        candidates.append(["-c:v", "hevc_qsv", "-global_quality", "24"])
        candidates.append(["-c:v", "av1_nvenc", "-preset", "p1", "-cq", "36"])
        candidates.append(["-c:v", "av1_amf", "-qp_i", "34", "-qp_p", "34"])
        candidates.append(["-c:v", "libsvtav1", "-crf", "34"])
        candidates.append(["-c:v", "hevc_nvenc", "-preset", "p1", "-cq", "30"])
        candidates.append(["-c:v", "hevc_amf", "-qp_i", "28", "-qp_p", "28"])
        candidates.append(["-c:v", "h264_qsv", "-global_quality", "24"])
        candidates.append(["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "30"])
        candidates.append(["-c:v", "h264_amf", "-qp_i", "28", "-qp_p", "28"])
        candidates.append(["-c:v", "libx265", "-crf", "25"])

        for enc_args in candidates:
            if _probe_encoder(enc_args):
                return enc_args
        # Software fallback
        return ["-c:v", "libx264", "-crf", "24"]

    encoder_args = _pick_encoder()
    print(f'[Info] Encoder: {" ".join(encoder_args)}')

    # Determine read dimensions and inference output based on mode
    if shrink_size and speed_target > 0 and max(w, h) > speed_target:
        # Speed mode: shrink input to fit within target tile size
        sf = speed_target / max(w, h)
        read_w = round(w * sf)
        read_h = round(h * sf)
        read_w -= read_w % 2  # ensure even
        read_h -= read_h % 2
        reader_vf = ["-vf", f"scale={read_w}:{read_h}:flags=lanczos,fps={fps}"]
        print(
            f"[Info] Speed mode: {w}x{h} -> {read_w}x{read_h} (target {speed_target}x{speed_target})"
        )
    else:
        read_w, read_h = w, h
        reader_vf = ["-vf", f"fps={fps}"]

    infer_w = read_w * model_scale
    infer_h = read_h * model_scale
    if infer_w != tw or infer_h != th:
        write_w, write_h = infer_w, infer_h
        scale_filter = ["-vf", f"scale={tw}:{th}:flags=lanczos"]
        print(
            f"[Info] Inference output {write_w}x{write_h} -> ffmpeg scale to {tw}x{th}"
        )
    else:
        write_w, write_h = tw, th
        scale_filter = []

    # Print tiling info
    native_in = processor.native_input_size
    _tile_dims = None
    if native_in and (native_in[0] != read_w or native_in[1] != read_h):
        _tile_dims = native_in  # CoreML fixed model: (w, h)
    elif processor.tile > 0:
        _tile_dims = (
            processor.tile,
            processor.tile,
        )  # explicit tile for ONNX/ncnn/dynamic CoreML
    if _tile_dims:
        ti_w, ti_h = _tile_dims
        p_h = (ti_h - read_h % ti_h) % ti_h
        p_w = (ti_w - read_w % ti_w) % ti_w
        n_tiles = ((read_h + p_h) // ti_h) * ((read_w + p_w) // ti_w)
        print(
            f"[Info] Tiling: {(read_w + p_w) // ti_w}x{(read_h + p_h) // ti_h} = {n_tiles} tiles of {ti_w}x{ti_h} per frame"
        )

    # ffmpeg writer command expects raw rgb24 stdin
    writer_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{write_w}x{write_h}",
        "-r",
        str(fps),
        "-i",
        "-",
        *encoder_args,
        *scale_filter,
        "-pix_fmt",
        "yuv420p",
        str(tmp_vid),
    ]

    # ffmpeg reader to emit rawvideo
    # Use -t to limit input duration, preventing reading past declared end
    # (some files have garbage/duplicate data beyond the stated duration)
    duration_limit = ["-t", str(duration)] if duration else []
    reader_cmd = [
        "ffmpeg",
        *duration_limit,
        "-i",
        str(input_path),
        *reader_vf,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-map",
        "0:v:0",
        "-hide_banner",
        "-loglevel",
        "error",
        "-",
    ]

    # start writer (capture ffmpeg stderr for diagnostics)
    wproc = subprocess.Popen(
        writer_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # start reader (capture ffmpeg stderr for diagnostics)
    rproc = subprocess.Popen(reader_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # helper to drain stderr to lists for later reporting
    w_stderr_lines = []
    r_stderr_lines = []

    def _drain_stream(stream, sink_list):
        try:
            for line in iter(stream.readline, b""):
                try:
                    sink_list.append(line.decode("utf-8", errors="replace"))
                except Exception:
                    sink_list.append(str(line))
        finally:
            try:
                stream.close()
            except Exception:
                pass

    # start background stderr readers
    serr_thread_r = threading.Thread(
        target=_drain_stream, args=(rproc.stderr, r_stderr_lines), daemon=True
    )
    serr_thread_w = threading.Thread(
        target=_drain_stream, args=(wproc.stderr, w_stderr_lines), daemon=True
    )
    serr_thread_r.start()
    serr_thread_w.start()

    frame_size_in = read_w * read_h * 3
    # We'll run a 3-stage pipeline: reader -> inferer -> writer, using queues
    infer_queue = pyqueue.Queue(maxsize=8)
    write_queue = pyqueue.Queue(maxsize=8)

    read_count = 0
    done_count = 0
    batch_seq = 0

    times = {"read": 0.0, "infer": 0.0, "write": 0.0}

    stop_event = threading.Event()

    def reader_thread():
        nonlocal read_count, batch_seq
        batch = []
        try:
            while not stop_event.is_set():
                t0 = time.time()
                data = rproc.stdout.read(frame_size_in)
                t1 = time.time()
                times["read"] += t1 - t0
                if not data or len(data) < frame_size_in:
                    break
                arr = np.frombuffer(data, dtype=np.uint8)
                arr = arr.reshape((read_h, read_w, 3))
                batch.append(arr)
                read_count += 1
                if len(batch) >= batch_size:
                    start_idx = read_count - len(batch)
                    infer_queue.put((batch_seq, batch, start_idx))
                    batch_seq += 1
                    batch = []
            # flush remaining
            if batch and not stop_event.is_set():
                start_idx = read_count - len(batch)
                infer_queue.put((batch_seq, batch, start_idx))
                batch_seq += 1
        finally:
            # signal end
            infer_queue.put(None)

    def infer_thread():
        try:
            while not stop_event.is_set():
                item = infer_queue.get()
                if item is None:
                    write_queue.put(None)
                    break
                seq, batch, start_idx = item
                t0 = time.time()
                outs = processor.upscale_batch(batch, start_idx=start_idx)
                t1 = time.time()
                times["infer"] += t1 - t0
                write_queue.put((seq, outs))
        except Exception:
            write_queue.put(None)
            raise

    def writer_thread():
        nonlocal done_count
        try:
            while not stop_event.is_set():
                item = write_queue.get()
                if item is None:
                    break
                seq, outs = item
                for out in outs:
                    if stop_event.is_set():
                        break
                    t0w = time.time()
                    if out.dtype != np.uint8:
                        out = out.astype("uint8")
                    if out.shape[1] != write_w or out.shape[0] != write_h:
                        if Image is None:
                            raise RuntimeError(
                                "Pillow required to adjust output frames"
                            )
                        out = np.array(
                            Image.fromarray(out).resize(
                                (write_w, write_h), Image.LANCZOS
                            )
                        )
                    raw = out.tobytes()
                    # Write in chunks to avoid OSError on Windows pipes
                    mv = memoryview(raw)
                    chunk_size = 1 << 20  # 1 MB
                    for i in range(0, len(mv), chunk_size):
                        wproc.stdin.write(mv[i : i + chunk_size])
                    t1w = time.time()
                    times["write"] += t1w - t0w
                    done_count += 1
                    _print_progress(done_count, nb_frames)
        except (OSError, BrokenPipeError):
            # ffmpeg process died (e.g. killed by Ctrl+C)
            stop_event.set()
        finally:
            try:
                wproc.stdin.close()
            except Exception:
                pass

    # start threads
    rthread = threading.Thread(target=reader_thread, daemon=True)
    ithread = threading.Thread(target=infer_thread, daemon=True)
    wthread = threading.Thread(target=writer_thread, daemon=True)

    rthread.start()
    ithread.start()
    wthread.start()

    def _cleanup_abort():
        """Kill subprocesses and signal threads to stop."""
        stop_event.set()
        for proc in (rproc, wproc):
            try:
                proc.kill()
            except Exception:
                pass
        for q in (infer_queue, write_queue):
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:
                pass
        try:
            infer_queue.put_nowait(None)
        except Exception:
            pass
        try:
            write_queue.put_nowait(None)
        except Exception:
            pass

    try:
        # Poll-based join so KeyboardInterrupt and stop_event are both handled
        for th in (rthread, ithread, wthread):
            while th.is_alive():
                if stop_event.is_set():
                    _cleanup_abort()
                    break
                th.join(timeout=0.5)
        # If stop_event was set by a thread (e.g. writer pipe error), clean up
        if stop_event.is_set():
            _cleanup_abort()
            for th in (rthread, ithread, wthread):
                th.join(timeout=2.0)
            try:
                tmp_vid.unlink()
            except Exception:
                pass
            print("\nAborted.", file=sys.stderr)
            return None
    except KeyboardInterrupt:
        print("\n[Info] Interrupted, cleaning up...", file=sys.stderr)
        _cleanup_abort()
        for th in (rthread, ithread, wthread):
            th.join(timeout=2.0)
        # cleanup temp file
        try:
            tmp_vid.unlink()
        except Exception:
            pass
        raise

    # cleanup reader process
    try:
        rproc.stdout.close()
    except Exception:
        pass
    rproc.wait()
    try:
        # close writer stdin if not already closed
        try:
            if wproc.stdin:
                wproc.stdin.close()
        except Exception:
            pass
        wproc.wait()
    except Exception:
        pass

    # ensure stderr drainers finished
    serr_thread_r.join(timeout=1.0)
    serr_thread_w.join(timeout=1.0)

    # if ffmpeg reader/writer reported errors, include them in exception
    if rproc.returncode and rproc.returncode != 0:
        err = "".join(r_stderr_lines[-200:])
        raise RuntimeError(
            f"ffmpeg reader failed (returncode={rproc.returncode}):\n{err}"
        )
    if wproc.returncode and wproc.returncode != 0:
        err = "".join(w_stderr_lines[-200:])
        raise RuntimeError(
            f"ffmpeg writer failed (returncode={wproc.returncode}):\n{err}"
        )

    # remux audio if present
    final_out = (
        Path(output_path)
        if output_path
        else input_path.with_name(input_path.stem + suffix + input_path.suffix)
    )
    if has_audio:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(tmp_vid),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            str(final_out),
        ]
    else:
        # no audio, move tmp to final
        cmd = ["ffmpeg", "-y", "-i", str(tmp_vid), "-c", "copy", str(final_out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        # try falling back to re-encoding audio if copy failed (some containers/codecs don't allow stream copy)
        if has_audio:
            cmd2 = [
                "ffmpeg",
                "-y",
                "-i",
                str(tmp_vid),
                "-i",
                str(input_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                str(final_out),
            ]
            p2 = subprocess.run(cmd2, capture_output=True, text=True)
            if p2.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg final mux failed (copy and re-encode attempts):\nfirst:\n{p.stderr}\nsecond:\n{p2.stderr}\n"
                )
        else:
            raise RuntimeError(f"ffmpeg final mux failed: {p.stderr}\n{p.stdout}")

    # cleanup
    try:
        tmp_vid.unlink()
    except Exception:
        pass

    print("\n" + t("saved", path=final_out))
    total_io = times["read"] + times["infer"] + times["write"]
    print(
        f"Timing breakdown (s): read={times['read']:.2f}, infer={times['infer']:.2f}, write={times['write']:.2f}, total={total_io:.2f}"
    )
    if done_count:
        print(
            f"Per-frame (ms): read={(times['read']/done_count)*1000:.1f}, infer={(times['infer']/done_count)*1000:.1f}, write={(times['write']/done_count)*1000:.1f}"
        )
    return str(final_out)


def _diag_torch_directml_model(wrapper, sample, model_name=None):
    """Lightweight DirectML diagnostic stub.

    The detailed, crash-prone forward-trace diagnostic used during debugging
    has been disabled. If you need to run an in-depth DirectML module trace,
    use the standalone diagnostic scripts (e.g. bench_dml.py / diag_*.py)
    or re-enable a custom trace manually.
    """
    try:
        sys.stderr.write("[Diag] DirectML forward-trace disabled (lightweight stub).")
    except Exception:
        pass


def _print_progress(done: int, total: int, width: int = 30):
    if total <= 0:
        total = 1
    frac = min(1.0, done / total)
    filled = int(round(width * frac))
    bar = "[" + "#" * filled + "-" * (width - filled) + "]"
    pct = int(frac * 100)
    sys.stderr.write(f"\r{bar} {pct}% ({done}/{total})")
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output")
    parser.add_argument(
        "-d",
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps", "rocm", "directml", "vulkan"],
    )
    parser.add_argument("-m", "--model", default="auto")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("-n", "--dry-run", action="store_true")
    parser.add_argument(
        "-H",
        "--half",
        action="store_true",
        help="Use half precision (fp16) for PyTorch inference (no effect on CoreML)",
    )
    parser.add_argument(
        "-t",
        "--tile",
        type=int,
        default=None,
        help="Tile size for chunked inference (default: auto; -t 0 = no tiling; -t N = tile at NxN)",
    )
    parser.add_argument(
        "-c",
        "--compile",
        action="store_true",
        help="Use torch.compile() for PyTorch inference (no effect on CoreML)",
    )
    # channels_last option removed (no benefit observed on DirectML)
    parser.add_argument(
        "-e",
        "--backend",
        default="auto",
        choices=["auto", "coreml", "onnx", "ncnn", "pth", "pillow"],
        help="Inference backend (auto=best available: coreml>onnx>ncnn>pth>pillow)",
    )
    parser.add_argument(
        "-s",
        "--shrink-size",
        nargs="?",
        type=int,
        const=-1,
        default=0,
        help="Speed mode: shrink input then upscale. Optional value = shrink target size (omitted = auto). E.g. -s 512 -t 128 shrinks to 512 then tiles at 128",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if path.is_file():
        files = [path]
    else:
        files = find_input_files(path)
    if not files:
        print(t("no_input"))
        return 1

    device = args.device if args.device != "auto" else detect_best_device()
    print(t("device", device=device))

    # Processor cache: reuse across files with same (model, tile_size) key
    processor_cache = {} if not args.dry_run else None

    for f in files:
        print("\n" + "=" * 60)
        print(t("processing", path=f))
        try:
            start = time.time()
            out = run_one(
                f,
                Path(args.output) if args.output else None,
                device,
                args.model,
                args.batch_size,
                args.dry_run,
                tile=args.tile,
                half=args.half,
                compile_model=args.compile,
                # channels_last removed
                backend=args.backend,
                _processor_cache=processor_cache,
                shrink_size=args.shrink_size if args.shrink_size is not None else 0,
            )
            if not args.dry_run and out is not None:
                elapsed = time.time() - start
                print(f"\nProcessing time: {elapsed:.1f}s")
            if out is None:
                return 1
        except KeyboardInterrupt:
            print("\nAborted.", file=sys.stderr)
            return 130
        except Exception as e:
            print(f"Error processing {f}: {e}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
