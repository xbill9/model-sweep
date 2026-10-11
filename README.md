# model-sweep

Small open-weight models under vLLM, compared against Gemma 4 E2B on low-end accelerators: one TPU v5e chip
first, with the L4 (EC2 G6) and T4G (EC2 G5g) to follow. Every model here is Apache 2.0 or MIT, so a
checkpoint can be repacked and republished.

This tree sits outside `~/gemma4-dev` so that monorepo stays Gemma-only. It uses the same rig naming, with
the model family added to slot 4 (`NAMING.md`).

#### Layout

- `survey.py` reads license, architecture and parameter count from Hugging Face and writes `LINEUP.md`, with
  one table per hardware profile (`v5e1`, `g6`, `g5g`). Memory figures are the measured ones from
  `~/gemma4-dev`.
- `LINEUP.md` is generated. Do not hand-edit it; change `LINEUP` or `HARDWARE` in `survey.py` and rerun it.
- `NAMING.md` explains how slot 4 differs from `~/gemma4-dev/NAMING.md`.
- `<platform>-vllm-<hardware>-<family>_<size>/benchmarks/runs/` is where each finished arm is filed. Those
  directories are created when an arm is filed, the same way `~/gemma4-dev/jev-tpu-v5e1` files its arms.

#### Tiers on v5e

- **Tier 1** runs on a native JAX model in `tpu_inference`: Qwen3, Qwen2.5, Llama-architecture and Gemma 4
  checkpoints. Qwen3-4B and Gemma 4 E2B pass upstream's v6e tests.
- **Tier 2** runs vLLM's PyTorch model through the torchax fallback. Upstream does not test any of these on
  TPU, so each one needs to boot once before it is benchmarked. Qwen3.5 uses hybrid linear attention, which
  makes it the least likely to load.

Upstream has no v5e test matrix, only v6e and v7x. Every v5e result here is first-hand. On the GPUs every
architecture in the lineup has a CUDA model in vLLM, so the tiers apply to v5e only.

#### GPU notes

- **G6 (L4)** fits the whole lineup, including Qwen3-8B, which v5e and T4G cannot hold at 16-bit.
- **G5g (T4G)** has no bf16, so weights serve as fp16. It is also an aarch64 host, and vLLM does not publish a
  build for aarch64 + SM 7.5; `~/gemma4-dev/gpu-vllm-g5g-2b` carries the one that works. Check each model for
  fp16 overflow before benchmarking it.

#### Comparing against `2b`

Gemma 4 E2B is labelled 2B, but it holds 5.12B parameters and 9.54 GiB of 16-bit weights, because of its
per-layer embeddings. By resident bytes its nearest neighbour in the lineup is Qwen3-4B (7.49 GiB). By the
size on the label, it is Qwen3-1.7B and SmolLM2-1.7B. Report both comparisons.

#### Status

Nothing has provisioned. Next: port `~/gemma4-dev/jev-tpu-v5e1/tpu/run_quant.sh` (several arms on one
queued resource, then self-delete) to serve the v5e lineup, using the flex-start `v5litepod-1` path in
`us-west4-a`.
