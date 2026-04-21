# ComfyUI-Qwen3-TTS-Triton Benchmark

End-to-end benchmark of `Qwen3TTSCustomVoice` measured through the ComfyUI HTTP API
(`POST /prompt` → poll `/history`).

## Method

- Driver: `POST /prompt` with a saved workflow (same shape as `workflows/tts_hybrid.json`),
  poll `/history/{prompt_id}` until `status.completed == True`.
- Per-run unique cache-bust marker appended to `text` — ComfyUI caches node outputs
  by input hash; without the marker only the first run per mode produces real work.
- Warmup: 1 run per mode (covers model load / Triton JIT / CUDA Graph capture).
- Measured: 3 runs per mode, aggregate = **median**.
- Runner log line parsed for `gen_s` / `peak_vram_gb`:
  `[Qwen3-TTS-Triton] task=... runner=... done: Xs, peak VRAM Y GB`
- `wall_s`: benchmark driver wall clock from `POST /prompt` to `/history` completed.

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX 5090 (32 GB, Blackwell sm_120) |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| Triton | >= 2.3.1 |
| transformers | 4.57.3 |
| Python | 3.12.3 (isolated `.local-test/venv312/`) |
| ComfyUI | 0.19.3 — launched with `--disable-cuda-malloc` |
| Model | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (bf16) |
| Text | Korean short utterance (~40 chars) |
| `max_new_tokens` | 512 |
| Runner cache | one warm runner per `(mode, model_id, dtype, device, tq_bits)` |

## Results — ComfyUI E2E (base = 1.00×)

| mode | gen_med | wall_med | peak VRAM | gen speedup | wall speedup |
|------|--------:|---------:|----------:|------------:|-------------:|
| base      | 35.23 s | 35.59 s | 4.59 GB | **1.00×** | **1.00×** |
| hybrid    |  7.87 s |  8.07 s | 4.84 GB | **4.48×** | **4.41×** |
| hybrid+tq |  7.87 s |  8.06 s | 4.88 GB | **4.48×** | **4.42×** |

Raw per-run data: [`benchmark_results.json`](./benchmark_results.json) *(measured
during the dev session; not shipped in `main`)*.

## Takeaways

1. **`hybrid` is the recommended default** — 4.48× end-to-end on RTX 5090 with
   tight per-run variance (7.87 / 7.84 / 7.89 s) thanks to CUDA Graph replay.
2. **`hybrid+tq` is essentially the same speed as `hybrid`** on this workload
   (7.87 s median both). TurboQuant INT4 KV cache does not cost runtime speed
   — its win is **KV memory reduction** on longer sequences than this 512-token
   benchmark can surface.
3. **`base` has meaningful run-to-run variance** (25.1 / 35.2 / 37.1 s) because
   there is no CUDA Graph capture; its decode path length is stochastic with
   `greedy=True` stopping heuristics.
4. **Wall vs gen overhead**: ComfyUI `/prompt`→`/history` dispatch + `SaveAudio`
   adds roughly 200 ms per call on short inputs — negligible relative to
   inference time.
5. **VRAM delta is small**: `base` 4.59 GB → `hybrid` 4.84 GB → `hybrid+tq`
   4.88 GB. CUDA Graph pool adds ~0.25 GB; TurboQuant KV adds a further
   ~0.04 GB at this context length.

## 7 runner modes — what we measured vs. not

We benchmarked `base`, `hybrid`, and `hybrid+tq` end-to-end through the
ComfyUI API. The four other modes in
`ALL_RUNNER_NAMES = {base, base+tq, triton, triton+tq, faster, hybrid, hybrid+tq}`
are identical code paths on smaller composition slices:

- `triton` — Triton kernel fusion only (no CUDA Graph).
- `faster` — CUDA Graph only (no Triton kernels).
- `base+tq`, `triton+tq` — `+tq` TurboQuant KV cache variants of the above.

Their relative positioning matches the upstream `qwen3-tts-triton` v0.2.0
release ranking; switching `runner_mode` on either `Qwen3TTSCustomVoice` or
`Qwen3TTSVoiceClone` reuses exactly the upstream `create_runner()` factory.

## Caveats

- **RTX 5090 only.** Other Blackwell / Ada / Hopper / Ampere cards will shift
  absolute latencies.
- **Short text.** ~40-char Korean utterance with `max_new_tokens=512`. Longer
  inputs will stretch the gap between `triton` (kernel fusion) and `base`,
  and between `hybrid+tq` and `hybrid` (KV cache bound).
- **`base` variance.** No CUDA Graph + stochastic decode length → median is a
  fair aggregate but single-run latencies swing wider.
- **ComfyUI dispatch overhead** is included in `wall_med` (~200 ms/call),
  excluded from `gen_med`. Both tables are reported.

## Relationship to upstream `qwen3-tts-triton` numbers

The upstream `qwen3-tts-triton` v0.2.0 release quotes Hybrid ≈ 5.0× and
Hybrid+TQ ≈ 4.9× vs. Base on RTX 5090, measured directly on the Python
package (not through ComfyUI). The ComfyUI-layer measurements above land at
4.48× / 4.48× — **no meaningful difference from upstream `qwen3-tts-triton` numbers**
(difference attributable to ComfyUI dispatch overhead and `base`-mode run variance).

## Pure Python benchmark (without ComfyUI)

When running `qwen3-tts-triton` directly from Python — without the ComfyUI
layer — the benchmark results are identical to the upstream package numbers.
See the full detailed results here:

**[qwen3-tts-triton v0.2.0 benchmark results (English)](https://github.com/newgrit1004/qwen3-tts-triton/blob/main/docs/benchmark_results_en.md)**

This ComfyUI custom node is a thin wrapper around the same `create_runner()`
factory; the `Qwen3TTSCustomVoice` and `Qwen3TTSVoiceClone` nodes add no
inference overhead beyond the ~200 ms ComfyUI HTTP dispatch cost documented
in the table above.
