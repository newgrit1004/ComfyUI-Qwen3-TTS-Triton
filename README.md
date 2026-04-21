# ComfyUI-Qwen3-TTS-Triton

**Fast Qwen3-TTS for ComfyUI** — two nodes (Custom Voice + Voice Clone) wrapping [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) with Triton kernel fusion, CUDA Graph capture, and TurboQuant KV cache. **Up to 5.0× faster** than the unoptimised baseline on RTX 5090.

[![Qwen3-TTS Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-Qwen3--TTS--12Hz--1.7B-blue)](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
[![Upstream](https://img.shields.io/badge/GitHub-qwen3--tts--triton-black)](https://github.com/newgrit1004/qwen3-tts-triton)
[![Sibling](https://img.shields.io/badge/Sibling-ComfyUI--Omnivoice--Triton-black)](https://github.com/newgrit1004/ComfyUI-Omnivoice-Triton)
[![Sibling](https://img.shields.io/badge/Sibling-ComfyUI--ZImage--Triton-black)](https://github.com/newgrit1004/ComfyUI-ZImage-Triton)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE)

[한국어 README](README_KO.md)

## Features

- **~5.0× end-to-end speedup** — `hybrid` (Triton + CUDA Graph) vs unoptimised `base` on Qwen3-TTS 1.7B (RTX 5090, bf16). See [`benchmark/BENCHMARK.md`](benchmark/BENCHMARK.md).
- **7 runner modes** — `base`, `triton`, `faster`, `hybrid`, plus **TurboQuant KV cache** variants `base+tq`, `triton+tq`, `hybrid+tq` (INT4 KV, ~4× memory reduction).
- **Two dedicated nodes** — `Qwen3TTSCustomVoice` (speaker + instruct) and `Qwen3TTSVoiceClone` (reference AUDIO + optional transcript) — minimal required inputs, advanced params in optional.
- **Warm runner cache** keyed by `(runner_mode, model_id, dtype, device, tq_bits)` — no redundant `load_model()` calls between runs with the same config.
- **Voice-clone prompt precomputation** — ComfyUI AUDIO is downmixed to mono, x-vector extracted once, no upstream file-loader round-trip.
- **Node-local vendor isolation** — install all deps into `vendor/` with `--no-deps` so this node does not fight other ComfyUI nodes over `transformers` / `accelerate` versions.
- **Torch-protective `install.py`** — never replaces ComfyUI's CUDA torch with a CPU-only wheel.

## Installation

### Method 1: ComfyUI Manager (recommended)

Search for **"Qwen3 Triton TTS"** in ComfyUI Manager and click **Install**.

### Method 2: Manual install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/newgrit1004/ComfyUI-Qwen3-TTS-Triton.git
cd ComfyUI-Qwen3-TTS-Triton
python install.py          # run from the ComfyUI venv
```

### Method 3: Vendor-isolated install (avoid cross-node dep conflicts)

```bash
python -m pip install --upgrade --no-deps --ignore-requires-python \
  --target ComfyUI/custom_nodes/ComfyUI-Qwen3-TTS-Triton/vendor \
  qwen3-tts-triton==0.2.0 \
  qwen-tts==0.1.1 \
  faster-qwen3-tts==0.2.5 \
  transformers==4.57.3 \
  accelerate==1.12.0 \
  huggingface-hub==0.36.2
```

The node prepends `vendor/` to `sys.path`, so these packages override the shared ComfyUI environment **only for this node**.

### Why `--no-deps`?

`qwen3-tts-triton` depends on `qwen-tts` and `faster-qwen3-tts`, each declaring their own torch constraints. On some systems pip's resolver will happily install a CPU-only `torch` wheel that satisfies these constraints, silently replacing ComfyUI's CUDA-enabled build and breaking GPU acceleration.

`install.py` installs the three core packages with `--no-deps` and then pulls in the remaining torch-independent dependencies (`transformers>=4.57.0`, `accelerate`, `huggingface-hub`, `triton`, `soundfile`, etc.) explicitly. Your ComfyUI `torch` is never modified.

If torch does get broken by another package:

```bash
pip install --upgrade --force-reinstall torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

### Requirements

- **Python ≥ 3.12** (`qwen3-tts-triton` pins `requires-python>=3.12`)
- **PyTorch with CUDA 12.8+**, Blackwell / Ada / Hopper / Ampere GPU
- **ComfyUI launched with `--disable-cuda-malloc`** — required for `hybrid` / `+tq` modes. See *Running ComfyUI* below.

### Local checkout fallback (developer mode)

If you want to develop against a local `qwen3-tts-triton` checkout instead of the PyPI wheel:

```bash
export COMFYUI_QWEN3_TTS_TRITON_SRC=/path/to/qwen3-tts-triton
```

The env var can point at the repo root or its `src/` directory. The node prepends that path to `sys.path` before importing `qwen3_tts_triton`.

## Nodes

<details>
<summary><strong>Qwen3 TTS (Custom Voice)</strong> — text → <code>AUDIO</code></summary>

**Required inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | STRING, multiline | sample | Text to synthesize. |
| `runner_mode` | enum | `hybrid` | `base`, `base+tq`, `triton`, `triton+tq`, `faster`, `hybrid`, `hybrid+tq`. |
| `language` | STRING | `English` | Language name passed to the upstream runner. |
| `speaker` | STRING | `vivian` | Speaker id. |

**Optional inputs** (advanced — defaults work for most cases):

| Parameter | Default | Description |
|---|---|---|
| `instruct` | empty | Style instruction string. |
| `model_id` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | HuggingFace id or local path. |
| `dtype` | `bf16` | `bf16`, `fp16`, `fp32`. |
| `device` | `cuda` | CUDA only. |
| `tq_bits` | 4 | TurboQuant bits for `+tq` modes (3 or 4). |
| `temperature` | 0.9 | Sampling temperature (0–2). |
| `top_k` | 50 | Top-k sampling (1–512). |
| `repetition_penalty` | 1.05 | Repetition penalty (1–2). |
| `max_new_tokens` | 2048 | Max decoder tokens (64–8192). |
| `greedy` | False | Force deterministic decode. |

**Output:** `audio` (AUDIO) — `{"waveform": tensor[1, 1, T], "sample_rate": int}`.

</details>

<details>
<summary><strong>Qwen3 TTS (Voice Clone)</strong> — text + reference AUDIO → <code>AUDIO</code></summary>

**Required inputs:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `text` | STRING, multiline | sample | Text to synthesize in the cloned voice. |
| `ref_audio` | AUDIO | — | Reference audio clip to clone. |
| `runner_mode` | enum | `hybrid` | Same 7 modes as Custom Voice. |
| `language` | STRING | `English` | Language name. |

**Optional inputs:**

| Parameter | Default | Description |
|---|---|---|
| `ref_text` | empty | Transcript of `ref_audio` (improves accuracy). |
| `xvec_only` | True | Use x-vector-only mode (skip audio token path). |
| `model_id` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | HuggingFace id or local path. |
| `dtype` | `bf16` | `bf16`, `fp16`, `fp32`. |
| `device` | `cuda` | CUDA only. |
| `tq_bits` | 4 | TurboQuant bits for `+tq` modes. |
| `temperature` | 0.9 | Sampling temperature. |
| `top_k` | 50 | Top-k sampling. |
| `repetition_penalty` | 1.05 | Repetition penalty. |
| `max_new_tokens` | 2048 | Max decoder tokens. |
| `greedy` | False | Force deterministic decode. |

The node downmixes the reference AUDIO to mono and precomputes the voice-clone prompt (x-vector extraction) before calling the upstream runner — no file-path round-trip.

**Output:** `audio` (AUDIO) — `{"waveform": tensor[1, 1, T], "sample_rate": int}`.

</details>

### Runner modes

| mode | what it does | when to use |
|---|---|---|
| `base` | reference Qwen3-TTS runner, no optimisation | sanity check / quality reference |
| `triton` | fused Triton kernels (RMSNorm, SwiGLU, M-RoPE) | long text, Triton-friendly GPUs |
| `faster` | CUDA Graph capture via `faster-qwen3-tts` | short repeated calls |
| `hybrid` | Triton fusion **and** CUDA Graph capture — **recommended** | default production |
| `base+tq` | `TurboQuantKVCache` (INT4) only | KV-memory-bound workloads |
| `triton+tq` | Triton + TurboQuant KV cache | memory + throughput balance |
| `hybrid+tq` | Triton + CUDA Graph + TurboQuant KV cache | **max memory savings at near-max speed** |

`tq_bits` controls TurboQuant bits on `+tq` modes (3 or 4; 4 recommended).

### Runner caching

The runner is cached at module scope, keyed by `(runner_mode, model_id, dtype, device, tq_bits)`. A cache hit reuses the warm runner (no `load_model()` cost). A key change evicts the previous runner via `unload_model()` before building a new one, so VRAM never doubles during mode swaps.

### CUDA Graph first-run cost (`faster` / `hybrid` / `hybrid+tq`)

These modes capture a CUDA Graph on the **first** inference for a given input shape. Capture adds ~0.1–0.5 s to that first run. Subsequent runs with the same shape replay the graph and are consistently fast.

## Running ComfyUI: `--disable-cuda-malloc` is required

```bash
python ComfyUI/main.py --listen 0.0.0.0 --port 8188 --disable-cuda-malloc
```

Without `--disable-cuda-malloc`, ComfyUI's default `cudaMallocAsync` allocator conflicts with both `transformers`'s parallel shard loading and CUDA Graph capture/teardown, causing fake OOMs and `hybrid` / `+tq` mode crashes.

## Example workflows

Four saved ComfyUI API workflows live in [`workflows/`](workflows/):

| Workflow | Node | Mode |
|---|---|---|
| [`tts_custom_voice_hybrid.json`](workflows/tts_custom_voice_hybrid.json) | `Qwen3TTSCustomVoice` | `hybrid` |
| [`tts_custom_voice_hybrid_tq.json`](workflows/tts_custom_voice_hybrid_tq.json) | `Qwen3TTSCustomVoice` | `hybrid+tq` |
| [`tts_voiceclone.json`](workflows/tts_voiceclone.json) | `Qwen3TTSVoiceClone` | `hybrid` |
| [`tts_voiceclone_tq.json`](workflows/tts_voiceclone_tq.json) | `Qwen3TTSVoiceClone` | `hybrid+tq` |

For other runner modes, load any workflow and change the `runner_mode` widget.

## Benchmark

Upstream v0.2.0 headline numbers (RTX 5090, cited — not measured at the ComfyUI layer):

| mode | gen speedup vs base |
|------|---:|
| base     | **1.00×** |
| triton   | ~1.4× |
| faster   | ~3.2× |
| hybrid   | **~5.0×** |
| hybrid+tq | **~4.9×** (with ~4× KV memory reduction) |

Full table + caveats: [`benchmark/BENCHMARK.md`](benchmark/BENCHMARK.md).

## References

- **Wrapped package:** [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton)
- **Upstream model:** [Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
- **Sibling Triton-accelerated nodes:**
  - [ComfyUI-Omnivoice-Triton](https://github.com/newgrit1004/ComfyUI-Omnivoice-Triton)
  - [ComfyUI-ZImage-Triton](https://github.com/newgrit1004/ComfyUI-ZImage-Triton)

## License

Apache-2.0 — see [LICENSE](LICENSE).
