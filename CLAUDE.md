# CLAUDE.md

A sweep of small Apache-2.0 / MIT open-weight models under vLLM on low-end accelerators (TPU v5e-1 first, then
EC2 G6 and G5g), with Gemma 4 E2B as the baseline. Read `README.md` first.

- **Naming:** `NAMING.md` here, on top of `~/gemma4-dev/NAMING.md`. Slot 4 is `<family>_<size>`, always with
  the family, including `gemma4_2b`. Add a family token to `NAMING.md` before naming a rig in a new family.
- **The lineup comes from code.** `survey.py` computes license, path, weight size and KV room, and writes
  `LINEUP.md`. Never type those numbers in by hand or hand-edit `LINEUP.md`.
- **Root references live in `~/gemma4-dev`:** `HARDWARE.md` (memory and numeric formats per part), `QUANTIZATION.md` (what
  `tpu_inference` supports), `RIG-ANALYSIS.md` (the worksheet to fill in before provisioning). Facts about
  non-Gemma checkpoints go in this tree, not in `~/gemma4-dev/MODELS.md`.
- **GPU rigs:** read `~/gemma4-dev/gpu-vllm-g6-2b` and `~/gemma4-dev/gpu-vllm-g5g-2b` before building one here.
  T4G has no bf16 and needs a custom aarch64 vLLM build (`~/gemma4-dev/HARDWARE.md`, the T4G section).
- **v5e provisions only through the Cloud TPU API**, flex-start `v5litepod-1` only in `us-west4-a`
  (`~/gemma4-dev/CLAUDE.md`, "Cloud gotchas"). Don't destroy a queued resource unless asked.
- **Never pair a number across images.** Every arm in one comparison runs on the same `vllm-tpu` digest; record
  the digest with each run. `nightly` moves.
- Use the system `python3`; never create a virtualenv. Tests are `unittest`.
