import argparse
import os
import sys
import tempfile
import shutil
import torch
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# repo root is two levels up from tools/model
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Build helper moved into the `archs` package; re-export for backwards compatibility
from archs import build_net

# load_weights moved to backends.pytorch to separate backend logic
from backends.pytorch import load_weights, find_checkpoint as pytorch_find_checkpoint


def find_checkpoint(model_name: str) -> Optional[str]:
    """Locate the PyTorch checkpoint by delegating to the PyTorch backend finder.

    Export only needs the original PyTorch checkpoint, so call the backend
    finder with just the model name and models_root.
    """
    return pytorch_find_checkpoint(
        model_name, models_root=os.path.join(REPO_ROOT, "models")
    )


def prepare_net(model_name: str, ckpt_path: str):
    try:
        net = build_net(model_name)
    except Exception as e:
        print(f"[Error] failed to build network: {e}", file=sys.stderr)
        return None
    try:
        load_weights(net, ckpt_path)
    except Exception as e:
        print(f"[Error] failed to load checkpoint weights: {e}", file=sys.stderr)
        return None

    net.eval()

    return net.cpu()


def validate_coreml(output_path: str) -> bool:
    try:
        import coremltools as ct
    except Exception as e:
        print("[Warn] coremltools not available for validation:", e)
        return False
    try:
        ct.models.MLModel(output_path)
        print("[Info] CoreML validation succeeded:", output_path)
        return True
    except Exception as e:
        print("[Error] CoreML validation failed:", e)
        return False


def validate_ncnn(output_path: str) -> bool:
    b = output_path
    p = os.path.splitext(output_path)[0] + ".param"
    ok = (
        os.path.exists(p)
        and os.path.exists(b)
        and os.path.getsize(p) > 0
        and os.path.getsize(b) > 0
    )
    if ok:
        print("[Info] NCNN validation succeeded:", p, b)
    else:
        print("[Error] NCNN validation failed: missing or empty files")
    return ok


def validate_onnx(output_path: str) -> bool:
    try:
        import onnx
    except Exception as e:
        print("[Warn] onnx is not installed, skipping validation:", e)
        return True

    try:
        model = onnx.load(output_path)
        onnx.checker.check_model(model)
        print("[Info] ONNX validation succeeded:", output_path)
        return True
    except Exception as e:
        print("[Error] ONNX validation failed:", e)
        return False


def quantize_onnx(onnx_path: str, out_path: str = None):
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except Exception as e:
        raise RuntimeError("onnxruntime.quantization not available: " + str(e))

    if out_path is None:
        base, ext = os.path.splitext(onnx_path)
        out_path = base + ".int8" + ext

    quantize_dynamic(onnx_path, out_path, weight_type=QuantType.QInt8)
    return out_path


def export_onnx(
    net_cpu: torch.nn.Module,
    output_path: str,
    input_size=(1, 3, 256, 256),
    dynamic: bool = False,
    fp16: bool = False,
    quantize: bool = False,
):

    # ensure output directory exists
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    dummy = torch.randn(*input_size)

    with torch.no_grad():
        kwargs = dict(
            export_params=True,
            do_constant_folding=True,
            opset_version=12,
            input_names=["input"],
            output_names=["output"],
            keep_initializers_as_inputs=False,
        )
        dynamo = False
        if dynamic:
            kwargs["opset_version"] = 18
            kwargs["dynamic_shapes"] = {"x": ["1~1024", 3, "1~4096", "1~4096"]}
            dynamo = True
        torch.onnx.export(net_cpu, dummy, output_path, **kwargs, dynamo=dynamo)

    # Simplify with onnx-simplifier and embed external data re-save.

    # 1) Ensure `onnx` is importable. If not, clean up generated artifacts
    #    and abort the ONNX post-processing step.
    try:
        import onnx
    except Exception as e:
        print("[Warn] onnx not available; cannot simplify or embed external data:", e)
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            ext = output_path + ".data"
            if os.path.exists(ext):
                os.remove(ext)
        except Exception as rm_e:
            print("[Warn] failed to remove generated ONNX artifacts:", rm_e)
        return

    # 2) Attempt to load the ONNX file with external data. If that fails,
    #    fall back to loading without external data and manually embed
    #    weights from the external file(s).
    model = None
    try:
        model = onnx.load(output_path, load_external_data=True)
    except Exception as load_exc:
        print(
            "[Warn] loading ONNX external data failed, attempting manual embed:",
            load_exc,
        )
        try:
            model = onnx.load(output_path, load_external_data=False)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("[Warn] failed to load ONNX model even without external data:", e)
            return

        base_dir = os.path.dirname(os.path.abspath(output_path))
        for tensor in list(model.graph.initializer):
            if not tensor.external_data:
                continue
            info = {kv.key: kv.value for kv in tensor.external_data}
            location = (
                info.get("location")
                or info.get("filename")
                or (os.path.basename(output_path) + ".data")
            )
            if not os.path.isabs(location):
                ext_path = os.path.join(base_dir, location)
            else:
                ext_path = location
            try:
                with open(ext_path, "rb") as f:
                    offset = (
                        int(info["offset"])
                        if "offset" in info and info["offset"]
                        else None
                    )
                    length = (
                        int(info["length"])
                        if "length" in info and info["length"]
                        else None
                    )
                    if offset is not None:
                        f.seek(offset)
                    data = f.read(length) if length is not None else f.read()
                tensor.raw_data = data
                del tensor.external_data[:]
                try:
                    tensor.data_location = 0
                except Exception:
                    pass
                print(
                    f"[Info] embedded external data for tensor {tensor.name} from {ext_path}"
                )
            except Exception as re:
                print("[Warn] failed to embed external data for", tensor.name, re)

    # 3) Run onnx-simplifier (if available) and re-save the model as a
    #    single-file ONNX. Keep simplifier failures non-fatal for the
    #    overall export flow.
    try:
        from onnxsim import simplify

        try:
            model, check = simplify(model)
            print(
                "[Info] onnx-simplifier applied"
                if check
                else "[Warn] onnx-simplifier check failed"
            )
        except Exception as sim_exc:
            import traceback

            traceback.print_exc()
            print("[Warn] onnx-simplifier failed during simplify():", sim_exc)
    except Exception as e:
        print("[Warn] onnx-simplifier not available; skipping simplify:", e)

    # If requested, convert model weights to FP16 using onnxconverter_common
    if fp16:
        try:
            from onnxconverter_common import float16 as oc_float16
            import warnings

            warnings.filterwarnings(
                "ignore", category=UserWarning, module="onnxconverter_common.float16"
            )
            # Prefer keeping IO types as-is (weights -> float16, IO stays float32)
            model = oc_float16.convert_float_to_float16(
                model,
                keep_io_types=True,
                min_positive_val=1e-7,
                max_finite_val=65504.0,
                node_block_list=None,
                check_fp16_ready=True,
            )
            print("[Info] Converted ONNX model weights to FP16 (keep IO types)")
        except Exception as e:
            print(
                "[Warn] onnxconverter_common.float16 conversion not available or failed:",
                e,
            )

    try:
        # Try to save model with embedded weights (no external data).
        onnx.save_model(model, output_path, save_as_external_data=False)
    except Exception:
        tmp_path = output_path + ".tmp"
        onnx.save_model(model, tmp_path, save_as_external_data=False)
        shutil.move(tmp_path, output_path)

    # attempt to remove any external data file left behind
    ext_data = output_path + ".data"
    if os.path.exists(ext_data):
        try:
            os.remove(ext_data)
        except Exception as e:
            import traceback

            traceback.print_exc()
            print("[Warn] failed to remove external data file:", e)

    # optional INT8 dynamic quantization using ONNX Runtime
    if quantize:
        try:
            quantized = quantize_onnx(output_path)
            print(f"[Info] Quantized ONNX written to: {quantized}")
        except Exception as e:
            print("[Error] ONNX quantization failed:", e)
            raise


def export_coreml(
    net_cpu: torch.nn.Module,
    output_path: str,
    input_size=(1, 3, 256, 256),
    dynamic: bool = False,
    fp16: bool = False,
    quantize: bool = False,
):
    try:
        import coremltools as ct
    except Exception as e:
        raise RuntimeError("coremltools not installed: " + str(e))

    # trace to torchscript first
    example = torch.randn(*input_size)
    traced = torch.jit.trace(net_cpu, example)
    precision = "Float16" if fp16 else "Float32"
    if dynamic:
        h_dim = ct.RangeDim(lower_bound=32, upper_bound=4096, default=input_size[2])
        w_dim = ct.RangeDim(lower_bound=32, upper_bound=4096, default=input_size[3])
        ct_shape = ct.Shape(shape=(1, 3, h_dim, w_dim))
        print(
            f"Converting traced model to Core ML ({precision}, dynamic H/W, this may take a while)"
        )
    else:
        ct_shape = input_size
        print(
            f"Converting traced model to Core ML ({precision}, fixed {input_size[2]}x{input_size[3]}, this may take a while)"
        )
    convert_kwargs = dict(
        inputs=[ct.TensorType(name="input", shape=ct_shape)],
        outputs=[ct.TensorType(name="output")],
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32,
    )

    mlmodel = ct.convert(traced, **convert_kwargs)

    if quantize:
        print(
            "[Info] Applying 8-bit KMeans Palettization for CoreML model compression (this may take a while)..."
        )
        import coremltools.optimize.coreml as cto

        try:
            import sklearn  # check scikit-learn

            mode = "kmeans"
        except ImportError:
            print(
                "[Info] scikit-learn not installed, k-means palettization failed for CoreML model compression."
            )
            print(
                '[Info] scikit-learn is required for k-means quantization. To install, run: "pip install scikit-learn".'
            )
            print("[Info] falling back to 'uniform' mode. Quality may decrease.")
            mode = "uniform"
        config = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(mode=mode, nbits=8)
        )
        compressed_mlmodel = cto.palettize_weights(mlmodel, config)
        mlmodel = compressed_mlmodel

    mlmodel.save(output_path)
    print(f"Saved CoreML model: {output_path} ({precision})")


def export_ncnn(
    net_cpu: torch.nn.Module,
    output_path: str,
    input_size=(1, 3, 256, 256),
    dynamic: bool = False,
    fp16: bool = False,
    quantize: bool = False,
) -> bool:
    if not dynamic:
        print(
            "[Warn] NCNN export does not support fixed input shapes; exporting with dynamic shape"
        )
        return False

    try:
        import pnnx
    except ImportError:
        print("[Error] pnnx not installed. pip install pnnx")
        return False

    dummy = torch.randn(*input_size)

    work_dir = tempfile.mkdtemp(prefix="pnnx_")
    output_dir = os.path.dirname(os.path.abspath(output_path))
    model_name = os.path.splitext(os.path.basename(output_path))[0]
    pt_path = os.path.join(work_dir, f"{model_name}.pt")
    param_out = os.path.splitext(output_path)[0] + ".param"
    bin_out = output_path

    try:
        ncnn_param = os.path.join(work_dir, f"{model_name}.ncnn.param")
        ncnn_bin = os.path.join(work_dir, f"{model_name}.ncnn.bin")
        from pathlib import Path

        pt_path_posix = Path(pt_path).as_posix()
        ncnn_param_posix = Path(ncnn_param).as_posix()
        ncnn_bin_posix = Path(ncnn_bin).as_posix()

        print(f"[Info] Exporting {model_name} via pnnx ... (input_size={input_size})")
        pnnx.export(
            net_cpu,
            pt_path_posix,
            dummy,
            ncnnparam=ncnn_param_posix,
            ncnnbin=ncnn_bin_posix,
            fp16=fp16,
        )

        if not os.path.exists(ncnn_param) or not os.path.exists(ncnn_bin):
            print("[Error] pnnx did not produce ncnn output files.")
            return False

        os.makedirs(output_dir, exist_ok=True)
        shutil.move(ncnn_param, param_out)
        shutil.move(ncnn_bin, bin_out)
        print(f"[Info] ncnn model exported:")
        print(f"       {param_out}")
        print(f"       {bin_out}")
        return True
    except Exception as e:
        print(f"[Error] pnnx export failed: {e}")
        return False
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def process(
    net_cpu: torch.nn.Module,
    name: str,
    ext: str,
    dir: str,
    export_fn,
    validate_fn,
    args,
) -> bool:
    out_base = os.path.join(dir, name)
    if args.input_size:
        size_dir = os.path.join(out_base, str(args.input_size))
        os.makedirs(size_dir, exist_ok=True)
        output_path = os.path.join(size_dir, f"{args.model}.{ext}")
        dynamic = False
        input_size = (1, 3, args.input_size, args.input_size)
    else:
        os.makedirs(out_base, exist_ok=True)
        output_path = os.path.join(out_base, f"{args.model}.{ext}")
        dynamic = True
        input_size = (1, 3, 256, 256)

    try:
        result = export_fn(
            net_cpu,
            output_path,
            input_size,
            dynamic,
            args.fp16,
            args.quantize_8bit,
        )
        # treat explicit False as failure for export functions that return a boolean
        if isinstance(result, bool) and not result:
            print(f"[Error] {name} export reported failure")
            return False
    except Exception as e:
        print(f"[Error] {name} export failed:", e)
        return False

    # optional validation step
    if args.validate and validate_fn is not None:
        try:
            ok = validate_fn(output_path)
            if not ok:
                print(f"[Error] {name} validation failed")
                return False
        except Exception as e:
            print(f"[Warn] {name} validation raised an exception:", e)
            return False

    print(f"[Info] Completed {name} export successfully.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Unified export tool for ONNX / CoreML / NCNN"
    )
    parser.add_argument(
        "backend",
        nargs="?",
        default="all",
        choices=["all", "onnx", "coreml", "ncnn"],
        help='Backend to export to; omit or use "all" to export all supported backends',
    )
    parser.add_argument(
        "--model", "-m", required=True, help="Model name, e.g. realesr-animevideov3"
    )
    parser.add_argument(
        "--ckpt", "-c", help="Path to .pth checkpoint. Default: auto-detect in models/"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Base output directory name under models/ (e.g. release_v1).",
    )
    parser.add_argument(
        "--input-size",
        "-s",
        type=int,
        metavar="SIZE",
        help="Fixed square input size (single integer). Omit for dynamic-size export.",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Use FP16 where supported (best-effort)"
    )
    parser.add_argument(
        "--quantize-8bit",
        "-q",
        action="store_true",
        help="ONNX: Performs INT8 quantization to reduce size and potentially speed up CPU/NPU inference.\nCoreML: Uses 8-bit KMeans Palettization to reduce model size by 50% while maintaining high visual fidelity on Apple Silicon.",
    )
    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="Validate exported artifacts when possible",
    )

    args = parser.parse_args()

    # Determine which backends to run
    if args.backend == "all":
        selected = ["onnx", "coreml", "ncnn"]
    else:
        selected = [args.backend]

    # NCNN supports fixed-size export inferred from the dummy tensor.
    # Keep NCNN selected even when --input-size is provided so we can
    # produce fixed-shape NCNN exports too.

    # Skip CoreML on non-macOS but don't error when doing an 'all' export
    if "coreml" in selected and sys.platform != "darwin":
        print("[Warn] CoreML export not supported on this platform; skipping CoreML")
        selected.remove("coreml")

    if not selected:
        print("[Error] No backends to export (after filtering).")
        sys.exit(1)

    # Resolve base models directory according to rules:
    # - If --output provided: use models/<output>
    # - Else if --ckpt provided explicitly: derive a subdir name from the ckpt and use models/<derived>
    # - Else: default to models/
    if args.output:
        base_models_dir = os.path.join(REPO_ROOT, "models", args.output)
    elif args.ckpt:
        ckpt_path = args.ckpt
        ckpt_dir = os.path.dirname(ckpt_path)
        # If ckpt is inside a directory named 'models' (e.g. 'models/foo.pth' -> dirname 'models'),
        # prefer the file stem; otherwise use the parent directory name as the output folder.
        if not ckpt_dir or os.path.basename(ckpt_dir) == "models":
            dir_name = os.path.splitext(os.path.basename(ckpt_path))[0]
        else:
            dir_name = os.path.basename(ckpt_dir)
        base_models_dir = os.path.join(REPO_ROOT, "models", dir_name)
    else:
        base_models_dir = os.path.join(REPO_ROOT, "models")

    os.makedirs(base_models_dir, exist_ok=True)

    # Find checkpoint once (use auto-detect only if user did not explicitly pass --ckpt)
    ckpt = args.ckpt or find_checkpoint(args.model)
    if not ckpt:
        print(
            f"[Error] Cannot find checkpoint for {args.model}. Use --ckpt to specify.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Prepare network once and reuse deep-copies for each backend to avoid repeated checkpoint loads.
    net = prepare_net(args.model, ckpt)
    if net is None:
        sys.exit(1)

    any_failures = False
    if "onnx" in selected:
        any_failures = any_failures or not process(
            net,
            "onnx",
            "onnx",
            base_models_dir,
            export_onnx,
            validate_onnx,
            args,
        )

    if "coreml" in selected:
        any_failures = any_failures or not process(
            net,
            "coreml",
            "mlpackage",
            base_models_dir,
            export_coreml,
            validate_coreml,
            args,
        )

    if "ncnn" in selected:
        any_failures = any_failures or not process(
            net,
            "ncnn",
            "bin",
            base_models_dir,
            export_ncnn,
            validate_ncnn,
            args,
        )

    if any_failures:
        print("[Warn] Some backends failed to export. Check logs above for details.")
    else:
        print("[Info] All requested backends exported successfully.")


if __name__ == "__main__":
    main()
