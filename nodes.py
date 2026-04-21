"""ComfyUI nodes for Qwen3-TTS via the qwen3-tts-triton package."""

from __future__ import annotations

import inspect
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

import sys as _sys
# NUMBA_DISABLE_JIT=1 prevents librosa's @guvectorize functions from compiling
# on Python 3.12+ with a working numba install — disable only on 3.11 and below
# where numba JIT can fail inside ComfyUI's restricted runtime.
if _sys.version_info < (3, 12):
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

try:
    from qwen3_tts_triton import ALL_RUNNER_NAMES, create_runner

    _IMPORT_ERROR: str | None = None
except ImportError as exc:
    _IMPORT_ERROR = str(exc)
    create_runner = None  # type: ignore[assignment]
    ALL_RUNNER_NAMES = [
        "base",
        "base+tq",
        "triton",
        "triton+tq",
        "faster",
        "hybrid",
        "hybrid+tq",
    ]

_DTYPE_CHOICES = ["bf16", "fp16", "fp32"]
_DEVICE_CHOICES = ["cuda"]
_DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
_LANGUAGE_CHOICES = [
    "auto",
    "english",
    "korean",
    "chinese",
    "japanese",
    "french",
    "german",
    "spanish",
    "italian",
    "portuguese",
    "russian",
]

_runner_cache: dict[tuple[str, str, str, str, int], Any] = {}
_cache_lock = threading.Lock()


def _get_or_create_runner(
    runner_mode: str,
    model_id: str,
    dtype: str,
    device: str,
    tq_bits: int,
) -> Any:
    """Return a warm runner, evicting the previous one when config changes."""
    if create_runner is None:
        raise ImportError(
            "qwen3-tts-triton is not installed. "
            "Run: pip install qwen3-tts-triton\n"
            "Or set COMFYUI_QWEN3_TTS_TRITON_SRC=/path/to/qwen3-tts-triton[/src]\n"
            f"Original error: {_IMPORT_ERROR}"
        )

    key = (runner_mode, model_id, dtype, device, tq_bits)
    with _cache_lock:
        if key in _runner_cache:
            logger.info("[Qwen3-TTS-Triton] cache hit: %s", key)
            return _runner_cache[key]

        for old_key, old_runner in list(_runner_cache.items()):
            logger.info("[Qwen3-TTS-Triton] evicting cached runner: %s", old_key)
            try:
                old_runner.unload_model()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[Qwen3-TTS-Triton] unload failed: %s", exc)
            del _runner_cache[old_key]

        try:
            import comfy.model_management as model_management  # type: ignore[import-not-found]

            model_management.unload_all_models()
            model_management.soft_empty_cache()
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("[Qwen3-TTS-Triton] ComfyUI model unload failed: %s", exc)

        runner = create_runner(
            runner_mode,
            device=device,
            model_id=model_id,
            dtype=dtype,
            tq_bits=tq_bits,
        )
        runner.load_model()
        _runner_cache[key] = runner
        return runner


def _to_list(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_to_list(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float))


def _ensure_waveform_3d(audio: Any) -> list[list[list[float]]]:
    data = _to_list(audio)
    if _is_number(data):
        return [[[float(data)]]]
    if not isinstance(data, (list, tuple)) or not data:
        raise ValueError("Qwen3-TTS-Triton: audio output is empty or invalid")

    first = data[0]
    if _is_number(first):
        return [[[float(item) for item in data]]]

    if isinstance(first, (list, tuple)) and first and _is_number(first[0]):
        return [[[float(item) for item in channel] for channel in data]]

    if (
        isinstance(first, (list, tuple))
        and first
        and isinstance(first[0], (list, tuple))
    ):
        return [
            [
                [float(sample) for sample in channel]
                for channel in batch
            ]
            for batch in data
        ]

    raise ValueError("Qwen3-TTS-Triton: unsupported audio tensor shape")


def _comfy_audio(audio: Any, sample_rate: int) -> dict[str, Any]:
    import torch

    waveform = torch.tensor(_ensure_waveform_3d(audio), dtype=torch.float32)
    return {"waveform": waveform, "sample_rate": int(sample_rate)}


def _mixdown_to_mono(audio: dict[str, Any]) -> list[float]:
    waveform = _to_list(audio["waveform"])
    if not isinstance(waveform, list) or not waveform:
        raise ValueError("Qwen3-TTS-Triton: ref_audio waveform is empty")

    if isinstance(waveform[0], list) and waveform[0] and isinstance(waveform[0][0], list):
        channels = waveform[0]
    elif isinstance(waveform[0], list):
        channels = waveform
    else:
        return [float(sample) for sample in waveform]

    channel_lists = [[float(sample) for sample in channel] for channel in channels]
    if not channel_lists or not channel_lists[0]:
        raise ValueError("Qwen3-TTS-Triton: ref_audio waveform is empty")

    length = min(len(channel) for channel in channel_lists)
    return [
        sum(channel[index] for channel in channel_lists) / len(channel_lists)
        for index in range(length)
    ]


def _clone_ref_audio_input(audio: dict[str, Any]) -> tuple[Any, int]:
    sample_rate = int(audio["sample_rate"])
    mono = _mixdown_to_mono(audio)

    try:
        import numpy as np

        return np.asarray(mono, dtype=np.float32), sample_rate
    except ImportError:
        return mono, sample_rate


def _run_voice_clone(
    runner: Any,
    *,
    text: str,
    language: str,
    ref_audio: dict[str, Any],
    ref_text: str,
    xvec_only: bool,
    temperature: float,
    top_k: int,
    repetition_penalty: float,
    max_new_tokens: int,
    greedy: bool,
) -> dict[str, Any]:
    import torch

    clone_model = runner._load_clone_model()
    signature = inspect.signature(clone_model.generate_voice_clone)
    clone_ref_audio = _clone_ref_audio_input(ref_audio)
    kwargs: dict[str, Any] = {}

    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    def supports_kwarg(name: str) -> bool:
        return accepts_var_kwargs or name in signature.parameters

    def maybe_set(name: str, value: Any) -> None:
        if supports_kwarg(name):
            kwargs[name] = value

    maybe_set("text", text)
    maybe_set("language", language)
    maybe_set("ref_text", ref_text)
    maybe_set("temperature", temperature)
    maybe_set("top_k", top_k)
    maybe_set("repetition_penalty", repetition_penalty)
    maybe_set("max_new_tokens", max_new_tokens)

    prompt_builder = clone_model
    if not hasattr(prompt_builder, "create_voice_clone_prompt") and hasattr(clone_model, "model"):
        prompt_builder = clone_model.model

    if hasattr(prompt_builder, "create_voice_clone_prompt") and supports_kwarg("voice_clone_prompt"):
        prompt_items = prompt_builder.create_voice_clone_prompt(
            ref_audio=clone_ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=xvec_only,
        )
        kwargs["voice_clone_prompt"] = prompt_items
    else:
        maybe_set("ref_audio", clone_ref_audio)
        if xvec_only:
            maybe_set("xvec_only", True)
            maybe_set("x_vector_only_mode", True)
    if greedy:
        maybe_set("temperature", 0.0)
        maybe_set("top_k", 1)
        if supports_kwarg("do_sample"):
            kwargs["do_sample"] = False

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    wavs, sample_rate = clone_model.generate_voice_clone(**kwargs)
    elapsed = time.perf_counter() - start

    peak_vram_gb = 0.0
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3

    return {
        "audio": wavs,
        "sample_rate": sample_rate,
        "time_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
    }


_OPTIONAL_GENERATION_PARAMS = {
    "temperature": (
        "FLOAT",
        {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.05},
    ),
    "top_k": ("INT", {"default": 50, "min": 1, "max": 512, "step": 1}),
    "repetition_penalty": (
        "FLOAT",
        {"default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01},
    ),
    "max_new_tokens": (
        "INT",
        {"default": 2048, "min": 64, "max": 8192, "step": 1},
    ),
    "greedy": ("BOOLEAN", {"default": False}),
    "model_id": ("STRING", {"default": _DEFAULT_MODEL_ID}),
    "dtype": (_DTYPE_CHOICES, {"default": "bf16"}),
    "device": (_DEVICE_CHOICES, {"default": "cuda"}),
    "tq_bits": ("INT", {"default": 4, "min": 3, "max": 4, "step": 1}),
}


class Qwen3TTSCustomVoice:
    """Qwen3-TTS custom voice synthesis with Triton-accelerated runner."""

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello, this is a Qwen3 Triton TTS test.",
                    },
                ),
                "runner_mode": (ALL_RUNNER_NAMES, {"default": "hybrid"}),
                "language": (_LANGUAGE_CHOICES, {"default": "english"}),
                "speaker": ("STRING", {"default": "vivian"}),
            },
            "optional": {
                "instruct": ("STRING", {"multiline": True, "default": ""}),
                **_OPTIONAL_GENERATION_PARAMS,
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS-Triton"
    DESCRIPTION = (
        "Qwen3-TTS custom voice synthesis using qwen3-tts-triton. "
        "Select a speaker and optional style instruction. "
        "Advanced generation and model parameters are in the optional inputs."
    )

    def generate(
        self,
        text: str,
        runner_mode: str,
        language: str = "english",
        speaker: str = "vivian",
        instruct: str = "",
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 2048,
        greedy: bool = False,
        model_id: str = _DEFAULT_MODEL_ID,
        dtype: str = "bf16",
        device: str = "cuda",
        tq_bits: int = 4,
    ) -> tuple[dict[str, Any]]:
        if not text or not text.strip():
            raise ValueError("Qwen3TTSCustomVoice: `text` must be non-empty")

        runner = _get_or_create_runner(runner_mode, model_id, dtype, device, tq_bits)
        result = runner.generate(
            text=text,
            language=language,
            speaker=speaker,
            instruct=instruct or None,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            greedy=greedy,
        )

        logger.info(
            "[Qwen3-TTS-Triton] task=custom_voice runner=%s done: %.2fs, peak VRAM %.2f GB",
            runner_mode,
            result.get("time_s", 0.0),
            result.get("peak_vram_gb", 0.0),
        )

        return (_comfy_audio(result["audio"], result["sample_rate"]),)


class Qwen3TTSVoiceClone:
    """Qwen3-TTS voice cloning with Triton-accelerated runner."""

    @classmethod
    def INPUT_TYPES(cls):  # noqa: N802
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Hello, this is a Qwen3 Triton TTS voice clone test.",
                    },
                ),
                "ref_audio": ("AUDIO",),
                "runner_mode": (ALL_RUNNER_NAMES, {"default": "hybrid"}),
                "language": (_LANGUAGE_CHOICES, {"default": "english"}),
            },
            "optional": {
                "ref_text": ("STRING", {"multiline": True, "default": ""}),
                "xvec_only": ("BOOLEAN", {"default": True}),
                **_OPTIONAL_GENERATION_PARAMS,
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "Qwen3-TTS-Triton"
    DESCRIPTION = (
        "Qwen3-TTS voice cloning using qwen3-tts-triton. "
        "Provide a reference audio clip to clone its voice characteristics. "
        "Advanced generation and model parameters are in the optional inputs."
    )

    def generate(
        self,
        text: str,
        ref_audio: dict[str, Any],
        runner_mode: str,
        language: str = "english",
        ref_text: str = "",
        xvec_only: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        max_new_tokens: int = 2048,
        greedy: bool = False,
        model_id: str = _DEFAULT_MODEL_ID,
        dtype: str = "bf16",
        device: str = "cuda",
        tq_bits: int = 4,
    ) -> tuple[dict[str, Any]]:
        if not text or not text.strip():
            raise ValueError("Qwen3TTSVoiceClone: `text` must be non-empty")

        runner = _get_or_create_runner(runner_mode, model_id, dtype, device, tq_bits)
        result = _run_voice_clone(
            runner,
            text=text,
            language=language,
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            greedy=greedy,
        )

        logger.info(
            "[Qwen3-TTS-Triton] task=voice_clone runner=%s done: %.2fs, peak VRAM %.2f GB",
            runner_mode,
            result.get("time_s", 0.0),
            result.get("peak_vram_gb", 0.0),
        )

        return (_comfy_audio(result["audio"], result["sample_rate"]),)
