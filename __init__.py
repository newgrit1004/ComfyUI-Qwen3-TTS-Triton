"""ComfyUI-Qwen3-TTS-Triton: Qwen3-TTS via qwen3-tts-triton."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _add_vendor_dir_if_present() -> None:
    vendor_dir = Path(__file__).resolve().parent / "vendor"
    if vendor_dir.is_dir():
        vendor = str(vendor_dir)
        if vendor in sys.path:
            sys.path.remove(vendor)
        sys.path.insert(0, vendor)
        logger.info("[ComfyUI-Qwen3-TTS-Triton] added vendor dir %s", vendor)


def _resolve_sideload_dir(raw_path: str) -> str:
    candidate = Path(raw_path).expanduser()
    direct_pkg = candidate / "qwen3_tts_triton"
    src_pkg = candidate / "src" / "qwen3_tts_triton"
    if direct_pkg.is_dir():
        return str(candidate)
    if src_pkg.is_dir():
        return str(candidate / "src")
    return ""


def _sideload_source_if_needed() -> None:
    """Allow pre-PyPI development against a local qwen3-tts-triton checkout."""
    try:
        import qwen3_tts_triton  # noqa: F401

        return
    except ImportError:
        pass

    raw_path = os.environ.get("COMFYUI_QWEN3_TTS_TRITON_SRC", "")
    src_dir = _resolve_sideload_dir(raw_path)
    if src_dir and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        logger.info("[ComfyUI-Qwen3-TTS-Triton] sideloaded qwen3_tts_triton from %s", src_dir)


_add_vendor_dir_if_present()
_sideload_source_if_needed()

try:
    from .nodes import Qwen3TTSCustomVoice, Qwen3TTSVoiceClone
    _import_ok = True
except ImportError as exc:
    logger.warning(
        "[ComfyUI-Qwen3-TTS-Triton] failed to import nodes: %s. "
        "Install qwen3-tts-triton or set COMFYUI_QWEN3_TTS_TRITON_SRC.",
        exc,
    )
    Qwen3TTSCustomVoice = None  # type: ignore[assignment, misc]
    Qwen3TTSVoiceClone = None  # type: ignore[assignment, misc]
    _import_ok = False

if _import_ok:
    NODE_CLASS_MAPPINGS = {
        "Qwen3TTSCustomVoice": Qwen3TTSCustomVoice,
        "Qwen3TTSVoiceClone": Qwen3TTSVoiceClone,
    }
    NODE_DISPLAY_NAME_MAPPINGS = {
        "Qwen3TTSCustomVoice": "Qwen3 TTS (Custom Voice)",
        "Qwen3TTSVoiceClone": "Qwen3 TTS (Voice Clone)",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
