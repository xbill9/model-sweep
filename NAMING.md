# Rig naming in this tree

This tree follows `~/gemma4-dev/NAMING.md` for every slot except slot 4. Read that file first: slots 1, 2, 3
and 5, the rule that the name is a claim about `tpu.env`, and the derived names (MCP server key, skill stem,
zone-status cache) all apply here as written. This file only records how slot 4 differs.

```
<platform>-<runtime>-<hardware>-<family>_<size>[-<encoding>]
tpu      - vllm    - v5e1      - qwen3_1p7b
```

## Slot 4 — model, with the family spelled out

In `~/gemma4-dev` slot 4 is a Gemma 4 size and the family is implied. This tree mixes families, so slot 4
always carries two tokens joined by one underscore: **`<family>_<size>`**. That includes Gemma, so the
baseline is `gemma4_2b`. A bare `2b` never appears here.

**Why an underscore:** the hyphen is the field separator, and joining the two tokens directly makes them
ambiguous (`qwen317b` could be Qwen3 1.7B or Qwen31 7B). `~/gemma4-dev/NAMING.md` already allows `_` inside
one slot for GGUF encodings (`q4_0`), so the character set is unchanged. Splitting a name on `-` still gives
exactly four or five slots.

**Family token:** the vendor's family name in lowercase, with punctuation and the vendor name dropped. Keep
the generation digits.

| Checkpoint family | Token |
| --- | --- |
| Gemma 4 | `gemma4` |
| Qwen3 | `qwen3` |
| Qwen3.5 | `qwen35` |
| Qwen2.5 | `qwen25` |
| DeepSeek-R1-Distill-Qwen | `r1distill` |
| SmolLM2 / SmolLM3 | `smollm2` / `smollm3` |
| Phi-4-mini | `phi4mini` |
| Granite 3.3 | `granite33` |
| OLMo 2 | `olmo2` |
| Ministral 3 | `ministral3` |

Add a row here before naming a rig in a new family.

**Size token:** the size printed in the checkpoint name, in lowercase, with the decimal point written as
`p`: `0p6b`, `1p7b`, `4b`. When the name carries no size (Phi-4-mini), use the size the model card states.
Gemma keeps its existing convention: E2B is `2b` and E4B is `4b`. The size token is a label, and the
parameter count used in any arithmetic comes from the checkpoint (`survey.py` reads it from Hugging Face). For
example, Gemma 4 E2B is labelled `2b` but holds 5.12B parameters.

**One checkpoint per name.** Slot 4 names a family and size, and `MODEL_NAME` in the rig's `tpu.env` names
the exact checkpoint. Where a family ships several instruction-tuned checkpoints at one size (Qwen3-4B and
Qwen3-4B-Instruct-2507), the rig's README says which one, and a second one is a second rig whose name needs a
decision recorded here first. Base weights and fine-tunes are out of scope, as in `~/gemma4-dev`.

## Slots 1 to 3

Unchanged from `~/gemma4-dev/NAMING.md`, which already covers every accelerator this tree targets:
`tpu-vllm-v5e1` (v5e has no Compute Engine path, so it is always `tpu`), `gpu-vllm-g6` (L4 on EC2) and
`gpu-vllm-g5g` (T4G on EC2; the slot is the EC2 instance family). Read that file before adding a platform or
a part.

## Harness and rigs

Same split as `~/gemma4-dev/jev-tpu-v5e1`: a sweep harness runs many checkpoints on one accelerator, and each
finished arm is filed in the rig named for its accelerator and exact checkpoint, under
`<rig>/benchmarks/runs/`. Those rigs hold measurements only (no `server.py`, no MCP server) until one is
promoted to a serving rig.
