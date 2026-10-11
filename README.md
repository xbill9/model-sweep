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
- `tpu/` is the v5e-1 sweep harness, ported from `~/gemma4-dev/jev-tpu-v5e1/tpu/run_quant.sh`:
  `launch.sh` bundles the committed tree and creates a flex-start queued resource; on the VM `startup.sh`
  runs `run_sweep.sh`, which serves each arm through `serve.sh`, measures it, uploads, and deletes the queued
  resource. `sweep.env` holds every cloud value and the pinned image. `arms-v5e1-tier1.txt` and
  `arms-v5e1-tier2.txt` are the two runs.
- `gen_eval.py` (GSM8K and BFCL simple, scored in code) and `tpu/load_client.py` (fixed-work throughput,
  long prompts) come from `~/gemma4-dev/jev-tpu-v5e1` at `b584cd5`, with `data/` pinned by hash in
  `data/gen_manifest.json`.
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

#### Method

Every arm in a run serves the same pinned image (`tpu/sweep.env`) with the same `--max-model-len 4096` and
`--max-num-seqs 16`, because attention cost on this chip follows `--max-model-len`. Each arm measures:

- **load:** 256 output tokens per request with `ignore_eos`, so every model does the same work, at 1, 4 and
  16 requests at once.
- **long:** the same with 1,024- and 3,072-token prompts, each with a unique prefix so the prefix cache never
  serves one.
- **gen:** GSM8K (1,319 problems, greedy, 2,048-token answer limit) and, where the model has a tool-call
  format vLLM parses, BFCL simple (400 records). Qwen3, Qwen3.5 and SmolLM3 run with thinking off
  (`enable_thinking: false`), so GSM8K compares answers without hidden reasoning. R1-Distill has no switch
  and reasons inside the same limit.

The model's Hugging Face commit is recorded per arm in `logs/<tag>.meta.json`.

#### Running it

```shell
tpu/launch.sh arms-v5e1-tier1.txt          # dry run: prints the bundle, the run and the create command
tpu/launch.sh arms-v5e1-tier1.txt --go     # uploads and creates the queued resource
```

Results and logs land in `gs://aisprint-491218-bucket/model-sweep/<run>/`. The queued resource deletes
itself after the last arm, or at the 4-hour limit. Launch tier 2 as a second run.

#### Status

Nothing has provisioned. `run_sweep.sh` has run end to end only against stubbed `gcloud`, `docker` and
metadata.
