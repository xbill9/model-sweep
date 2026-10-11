#!/usr/bin/env python3
"""Small-model lineup: license, architecture, size and fit per accelerator, read from the Hugging Face API.

Writes LINEUP.md with one table per hardware profile. Every number in it is computed here; nothing is typed in
by hand.

    python3 survey.py                  # every profile, writes LINEUP.md
    python3 survey.py --hardware g6    # one profile
    python3 survey.py --stdout         # prints instead
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import urllib.request

# Architectures with a native JAX model in tpu_inference (tpu_inference/models/common/model_loader.py,
# upstream main 5454f2b23, 2026-10-11). Everything else falls back to vLLM's PyTorch model through torchax.
JAX_NATIVE = {
    "DeepseekV3ForCausalLM", "Gemma4ForCausalLM", "Gemma4ForConditionalGeneration", "GptOssForCausalLM",
    "Llama4ForCausalLM", "Llama4ForConditionalGeneration", "LlamaForCausalLM", "Qwen2ForCausalLM",
    "Qwen3ForCausalLM", "Qwen2_5_VLForConditionalGeneration",
}
# Passing all three columns on the v6e nightly matrix (no v5e matrix exists).
V6E_PASSING = {"Qwen/Qwen3-4B", "google/gemma-4-E2B-it", "google/gemma-4-E4B-it"}
REPACK_LICENSES = {"apache-2.0", "mit"}

# Device memory as measured on the part (~/gemma4-dev/HARDWARE.md and the gpu-vllm-* rigs), not the spec sheet.
# prefix is slots 1-3 of the rig name (~/gemma4-dev/NAMING.md); dtype is what the weights are served in.
HARDWARE = {
    "v5e1": dict(prefix="tpu-vllm-v5e1", mem_gib=15.75, dtype="bf16", tpu=True,
                 source="v5litepod-1 total HBM under vLLM, HARDWARE.md"),
    "g6": dict(prefix="gpu-vllm-g6", mem_gib=23034 / 1024, dtype="bf16", tpu=False,
               source="L4, 23034 MiB from nvidia-smi, gpu-vllm-g6-2b"),
    "g5g": dict(prefix="gpu-vllm-g5g", mem_gib=14913 / 1024, dtype="fp16", tpu=False,
                source="T4G, 14913 MiB visible to torch, HARDWARE.md; SM 7.5 has no bf16, and aarch64 + SM 7.5 "
                       "needs the vLLM build gpu-vllm-g5g-2b carries"),
}
UTIL = 0.90     # --gpu-memory-utilization the small-model rigs serve with
MIN_KV_GIB = 2.0

# (slot 4, MODEL_NAME). Slot 4 follows NAMING.md.
LINEUP = [
    ("gemma4_2b", "google/gemma-4-E2B-it"),
    ("qwen3_0p6b", "Qwen/Qwen3-0.6B"),
    ("qwen3_1p7b", "Qwen/Qwen3-1.7B"),
    ("qwen3_4b", "Qwen/Qwen3-4B"),
    ("qwen25_0p5b", "Qwen/Qwen2.5-0.5B-Instruct"),
    ("qwen25_1p5b", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("r1distill_1p5b", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"),
    ("smollm2_1p7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),
    ("qwen35_0p8b", "Qwen/Qwen3.5-0.8B"),
    ("qwen35_2b", "Qwen/Qwen3.5-2B"),
    ("olmo2_1b", "allenai/OLMo-2-0425-1B-Instruct"),
    ("granite33_2b", "ibm-granite/granite-3.3-2b-instruct"),
    ("smollm3_3b", "HuggingFaceTB/SmolLM3-3B"),
    ("phi4mini_3p8b", "microsoft/Phi-4-mini-instruct"),
    ("ministral3_3b", "mistralai/Ministral-3-3B-Instruct-2512-BF16"),
    # Checked and excluded; kept so the reason stays visible.
    ("qwen25_3b", "Qwen/Qwen2.5-3B-Instruct"),
    ("qwen3_8b", "Qwen/Qwen3-8B"),
]


def fetch(model_id):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{model_id}", timeout=30) as r:
        d = json.load(r)
    lic = (d.get("cardData") or {}).get("license") or next(
        (t.split(":", 1)[1] for t in d.get("tags", []) if t.startswith("license:")), "?")
    arch = ((d.get("config") or {}).get("architectures") or ["?"])[0]
    params = (d.get("safetensors") or {}).get("total")
    return {"license": lic, "arch": arch, "params": params, "sha": d.get("sha", "")[:9]}


def assess(hw, slot4, model_id, info):
    budget = hw["mem_gib"] * UTIL
    gib = info["params"] * 2 / 2**30 if info["params"] else None  # bf16 and fp16 are both 2 bytes
    kv = budget - gib if gib is not None else None
    if hw["tpu"]:
        path = "JAX" if info["arch"] in JAX_NATIVE else "torchax"
    else:
        path = "CUDA"
    licensed = info["license"].lower() in REPACK_LICENSES
    fits = kv is not None and kv >= MIN_KV_GIB
    if not licensed:
        tier = "excluded: license"
    elif not fits:
        tier = "excluded: memory"
    elif path in ("JAX", "CUDA"):
        tier = "1"
    else:
        tier = "2"
    return dict(rig=f"{hw['prefix']}-{slot4}", model=model_id, path=path, licensed=licensed, gib=gib, kv=kv,
                tier=tier, v6e="✅" if hw["tpu"] and model_id in V6E_PASSING else "", **info)


def render(name, hw, rows):
    budget = hw["mem_gib"] * UTIL
    tiers = ("Tier 1 runs on a native JAX model, tier 2 on the torchax fallback. No v5e test matrix exists "
             "upstream; the v6e column marks models passing on v6e." if hw["tpu"] else
             "Every architecture here has a CUDA model in vLLM, so every model that fits is tier 1.")
    out = [
        f"## {name}: `{hw['prefix']}-*`",
        "",
        f"Memory {hw['mem_gib']:.2f} GiB ({hw['source']}) × {UTIL:g} = {budget:.2f} GiB. Weights at "
        f"{hw['dtype']}. A model fits when at least {MIN_KV_GIB:g} GiB is left for the KV cache. {tiers}",
        "",
        f"| Rig | MODEL_NAME | License | Architecture | Params (B) | {hw['dtype']} GiB | KV GiB | Path | v6e | Tier |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = {"1": 0, "2": 1}
    for r in sorted(rows, key=lambda r: (order.get(r["tier"], 2), r["params"] or 0)):
        out.append(
            f"| `{r['rig']}` | `{r['model']}` | {r['license']} | {r['arch']} | {r['params'] / 1e9:.2f} | "
            f"{r['gib']:.2f} | {r['kv']:.2f} | {r['path']} | {r['v6e']} | {r['tier']} |")
    counts = {t: sum(r["tier"] == t for r in rows) for t in ("1", "2")}
    out += ["", f"Tier 1: {counts['1']} models. Tier 2: {counts['2']} models. "
            f"Excluded: {len(rows) - counts['1'] - counts['2']}.", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hardware", choices=sorted(HARDWARE), action="append")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    with cf.ThreadPoolExecutor(8) as ex:
        infos = list(ex.map(fetch, [m for _, m in LINEUP]))
    sections = ["# Lineup", "",
                f"Generated by `survey.py` on {datetime.date.today().isoformat()} from the Hugging Face API.", ""]
    for name in args.hardware or HARDWARE:
        hw = HARDWARE[name]
        sections.append(render(name, hw, [assess(hw, s, m, i) for (s, m), i in zip(LINEUP, infos)]))
    text = "\n".join(sections)
    if args.stdout:
        print(text)
    else:
        with open("LINEUP.md", "w") as f:
            f.write(text)
        print(f"wrote LINEUP.md ({len(LINEUP)} models)")


if __name__ == "__main__":
    main()
