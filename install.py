"""Installation script for ComfyUI-Qwen3-TTS-Triton.

BEST PRACTICES (mirrors sibling ComfyUI-Omnivoice-Triton / ComfyUI-ZImage-Triton):
1. NEVER touch torch / torchaudio / torchvision — ComfyUI manages these.
2. Use --no-deps for any package that might pull in a torch version different
   from the one ComfyUI ships with.
3. Install packages individually for better error tracking.
4. Verify installation at the end.

WHY THIS EXISTS:
`qwen3-tts-triton` (>=0.2.0) depends on `qwen-tts` and `faster-qwen3-tts`,
which in turn declare `torch` constraints. A naive
`pip install qwen3-tts-triton` can replace ComfyUI's CUDA-enabled torch with
a CPU-only wheel depending on pip's resolver, breaking GPU acceleration. We
work around this by installing the core packages with `--no-deps` and then
pulling in the remaining, torch-independent dependencies explicitly.

Run from the ComfyUI venv (same Python that runs ComfyUI):

    python ComfyUI/custom_nodes/ComfyUI-Qwen3-TTS-Triton/install.py
"""

import importlib
import importlib.util
import subprocess
import sys


QWEN3_TTS_TRITON_VERSION = "0.2.0"
QWEN_TTS_VERSION = "0.1.1"
FASTER_QWEN3_TTS_VERSION = "0.2.5"
TRANSFORMERS_MIN = "4.57.0"


def run_cmd(cmd, timeout=300):
    """Run a command; return (success, stdout, stderr)."""
    print(f"[Qwen3-TTS-Triton] $ {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode == 0:
            return True, result.stdout, result.stderr
        print(f"[Qwen3-TTS-Triton] Command failed: {result.stderr.strip()[:500]}")
        return False, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"[Qwen3-TTS-Triton] Timeout after {timeout}s")
        return False, "", "Timeout"
    except Exception as exc:  # noqa: BLE001
        print(f"[Qwen3-TTS-Triton] Error: {exc}")
        return False, "", str(exc)


def is_installed(module_name):
    return importlib.util.find_spec(module_name) is not None


def pip_install(package, no_deps=False, upgrade=False, ignore_requires_python=False):
    """Install one package, preferring uv if available, else pip."""
    flags = []
    if no_deps:
        flags.append("--no-deps")
    if upgrade:
        flags.append("--upgrade")
    if ignore_requires_python:
        flags.append("--ignore-requires-python")

    uv_cmd = [sys.executable, "-m", "uv", "pip", "install", package] + flags
    ok, _, _ = run_cmd(uv_cmd)
    if ok:
        return True

    pip_cmd = [sys.executable, "-m", "pip", "install", package] + flags
    ok, _, _ = run_cmd(pip_cmd)
    return ok


def check_torch():
    """Return (version, has_gpu) for the active torch install, if any."""
    try:
        import torch  # noqa: PLC0415

        has_gpu = torch.cuda.is_available() or (
            hasattr(torch, "xpu") and torch.xpu.is_available()
        )
        return torch.__version__, has_gpu
    except ImportError:
        return None, False


def main():
    # Fast path: if everything already imports cleanly, skip.
    try:
        import qwen3_tts_triton  # noqa: F401, PLC0415
        import qwen_tts  # noqa: F401, PLC0415
        import faster_qwen3_tts  # noqa: F401, PLC0415
        import transformers  # noqa: PLC0415

        tv = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        if tv >= (4, 57):
            print(
                f"[Qwen3-TTS-Triton] All dependencies present "
                f"(qwen3-tts-triton={qwen3_tts_triton.__version__}, "
                f"transformers={transformers.__version__}). Skipping."
            )
            return
        print(
            f"[Qwen3-TTS-Triton] WARNING: transformers "
            f"{transformers.__version__} is too old (need >= {TRANSFORMERS_MIN}). "
            "Continuing with install — this may upgrade transformers and could "
            "break other custom nodes that pin it."
        )
    except (ImportError, ValueError, AttributeError):
        pass

    print("=" * 60)
    print("[Qwen3-TTS-Triton] Installation starting")
    print("=" * 60)

    # Step 1 — sanity-check torch, but never modify it.
    print("\n[Qwen3-TTS-Triton] Step 1: Checking PyTorch")
    torch_version, has_gpu = check_torch()
    if torch_version is None:
        print(
            "[Qwen3-TTS-Triton] ERROR: PyTorch is not installed. "
            "ComfyUI requires torch — install ComfyUI first."
        )
        return

    if has_gpu:
        print(f"[Qwen3-TTS-Triton] PyTorch {torch_version} with GPU — OK")
    else:
        print(
            f"[Qwen3-TTS-Triton] WARNING: PyTorch {torch_version} has no GPU. "
            "qwen3-tts-triton needs CUDA for the triton / hybrid / +tq modes."
        )

    # Step 2 — install core packages with --no-deps
    print(
        "\n[Qwen3-TTS-Triton] Step 2: Installing qwen3-tts-triton + qwen-tts + "
        "faster-qwen3-tts with --no-deps (to protect your PyTorch install)"
    )

    core_packages = [
        ("qwen_tts", f"qwen-tts=={QWEN_TTS_VERSION}"),
        ("faster_qwen3_tts", f"faster-qwen3-tts=={FASTER_QWEN3_TTS_VERSION}"),
        ("qwen3_tts_triton", f"qwen3-tts-triton=={QWEN3_TTS_TRITON_VERSION}"),
    ]

    for import_name, pip_spec in core_packages:
        if is_installed(import_name):
            # Force reinstall at the pinned version even if already present;
            # --no-deps ensures torch stays untouched.
            pass
        if not pip_install(pip_spec, no_deps=True, upgrade=True, ignore_requires_python=True):
            print(
                f"[Qwen3-TTS-Triton] ERROR: failed to install {pip_spec}. "
                f"Try manually: pip install {pip_spec} --no-deps --ignore-requires-python"
            )

    # Step 3 — install the remaining torch-independent dependencies.
    print("\n[Qwen3-TTS-Triton] Step 3: Installing remaining dependencies")

    # (import_name, pip_spec, label, no_deps?)
    extra_packages = [
        ("accelerate", "accelerate", "HF accelerate", False),
        ("huggingface_hub", "huggingface-hub", "HF hub client", False),
        ("transformers", f"transformers>={TRANSFORMERS_MIN}", "HF transformers", False),
        # triton pins specific torch ABIs; use --no-deps to avoid torch swap.
        ("triton", "triton>=2.3.1", "Triton kernels runtime", True),
        ("numpy", "numpy", "NumPy", True),
        ("soundfile", "soundfile", "Audio I/O", True),
        ("scipy", "scipy", "SciPy", True),
        ("pynvml", "pynvml", "NVML bindings (VRAM monitoring)", True),
        ("sentencepiece", "sentencepiece", "Tokenizer", True),
        ("plotly", "plotly", "Plotly (analysis UIs)", True),
    ]

    for import_name, pip_spec, label, no_deps in extra_packages:
        if is_installed(import_name):
            print(f"[Qwen3-TTS-Triton] {label} ({pip_spec}) — already present")
            continue
        print(f"[Qwen3-TTS-Triton] Installing {label} ({pip_spec})")
        ok = pip_install(pip_spec, no_deps=no_deps)
        if not ok:
            # Fallback: plain pip, no --no-deps.
            run_cmd([sys.executable, "-m", "pip", "install", pip_spec])

    # Step 4 — final verification
    print("\n" + "=" * 60)
    print("[Qwen3-TTS-Triton] Verification")
    print("=" * 60)

    importlib.invalidate_caches()

    verify = [
        ("qwen3_tts_triton", "qwen3-tts-triton"),
        ("qwen_tts", "qwen-tts"),
        ("faster_qwen3_tts", "faster-qwen3-tts"),
        ("transformers", "transformers"),
        ("triton", "triton"),
    ]
    for module_name, pip_name in verify:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            print(f"  [FAIL] {pip_name}: not found on disk")
            continue
        try:
            importlib.import_module(module_name)
            print(f"  [OK]   {pip_name}")
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [FAIL] {pip_name}: on disk but failed to import "
                f"({type(exc).__name__}: {exc})"
            )

    torch_version_after, has_gpu_after = check_torch()
    if torch_version_after and has_gpu_after:
        print(f"  [OK]   PyTorch {torch_version_after} (GPU)")
    elif torch_version_after:
        print(f"  [WARN] PyTorch {torch_version_after} (no GPU)")
    else:
        print("  [FAIL] PyTorch — not installed")

    if has_gpu and not has_gpu_after:
        print(
            "\n"
            + "=" * 60
            + "\n[Qwen3-TTS-Triton] WARNING: PyTorch lost its GPU support.\n"
            "Restore it with a CUDA wheel matching your driver, e.g.:\n"
            "  pip install --upgrade --force-reinstall \\\n"
            "    torch torchaudio \\\n"
            "    --index-url https://download.pytorch.org/whl/cu128\n"
            "Or see https://pytorch.org/get-started/locally/\n"
            + "=" * 60
        )


if __name__ == "__main__":
    main()
