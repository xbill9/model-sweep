---
title: "Small Apache-2.0 Models Against Gemma 4 E2B on One TPU v5e: Which Ones Keep Up?"
published: false
description: "Fourteen Apache-2.0 and MIT open-weight models that fit one TPU v5e chip, served one after another through vLLM on the same pinned image and scored on GSM8K and BFCL against Gemma 4 E2B. Qwen3-4B matches Gemma's accuracy at 0.57x its output, SmolLM3-3B and Granite 3.3 2B match it on tool calls, and Gemma keeps the lead on long prompts."
tags: tpu, vllm, gemma, llm
cover_image: https://raw.githubusercontent.com/xbill9/model-sweep/main/article/devto-cover.a42e27c5.jpg
---

This article provides a step by step guide to surveying small open-weight models on one Google Cloud TPU v5e chip through vLLM, with Gemma 4 E2B as the baseline. Every model is Apache-2.0 or MIT licensed, so any of them can be repacked and redistributed, and every one is served on the same pinned image with the same settings, then timed and scored by code.

Gemma 4 E2B is small and workable on a v5e-1, but its attention is unusual: two head sizes, 256 for sliding-window layers and 512 for full-attention layers, one KV head, and 20 of its 35 layers reading another layer's KV cache. The question here is which plainer model can stand in for it. Qwen3-4B matches Gemma on both accuracy tests at 0.57x the output. SmolLM3-3B and Granite 3.3 2B match it on tool calls at 0.76x and 0.93x. With long prompts, Gemma stays ahead of every model of comparable accuracy.

https://github.com/xbill9/model-sweep

---

#### Why One v5e Chip?

A `v5litepod-1` is the cheapest TPU Google Cloud rents and is usually available as flex-start capacity in `us-west4-a`. It has 15.75 GiB of HBM, and vLLM gets 0.9 of that, so a bf16 model has to leave room for its KV cache in about 14 GiB.

That rules the field down to roughly 4B parameters. Gemma 4 E2B loads in 8.97 GiB on this chip, which makes it one of the larger models in the lineup.

---

#### At This Point You Should Have…

- A Google Cloud project with the Cloud TPU API enabled and flex-start TPU v5e quota in `us-west4-a`
- A GCS bucket for results, and a Hugging Face token saved in Secret Manager as `hf-token`
- `gcloud` with the alpha components, and Python 3 with no extra packages
- A clone: `git clone https://github.com/xbill9/model-sweep`

---

#### Step 1 — Build the Lineup From the Hub

`survey.py` reads each candidate's license, architecture and weight size from the Hugging Face API and decides whether it fits the chip. A model is in when its license is Apache-2.0 or MIT and at least 2 GiB is left for the KV cache. vLLM on TPU runs some architectures as native JAX models and the rest through a PyTorch fallback, so the survey splits the lineup into two tiers:

```bash
python3 survey.py
```

```text
| `tpu-vllm-v5e1-qwen3_4b` | `Qwen/Qwen3-4B` | apache-2.0 | Qwen3ForCausalLM | 4.02 | 7.49 | 6.68 | JAX | ✅ | 1 |
| `tpu-vllm-v5e1-gemma4_2b` | `google/gemma-4-E2B-it` | apache-2.0 | Gemma4ForConditionalGeneration | 5.12 | 9.54 | 4.63 | JAX | ✅ | 1 |
| `tpu-vllm-v5e1-smollm3_3b` | `HuggingFaceTB/SmolLM3-3B` | apache-2.0 | SmolLM3ForCausalLM | 3.08 | 5.73 | 8.45 | torchax |  | 2 |
| `tpu-vllm-v5e1-qwen25_3b` | `Qwen/Qwen2.5-3B-Instruct` | other | Qwen2ForCausalLM | 3.09 | 5.75 | 8.43 | JAX |  | excluded: license |
| `tpu-vllm-v5e1-qwen3_8b` | `Qwen/Qwen3-8B` | apache-2.0 | Qwen3ForCausalLM | 8.19 | 15.26 | -1.08 | JAX |  | excluded: memory |

Tier 1: 8 models. Tier 2: 7 models. Excluded: 2.
```

Qwen2.5-3B is out on its license and Qwen3-8B on memory. Everything else goes into two arms files, one per tier.

---

#### Step 2 — Describe Each Model in One Line

An arms file lists one model per line: the checkpoint, its rig name, what to measure, and any serving flags it needs. The tool-call parser and the thinking switch differ by family:

```text
google/gemma-4-E2B-it=gemma4_2b=load+long+gen=parser=gemma4,mm=image+audio+video
Qwen/Qwen3-4B=qwen3_4b=load+long+gen=parser=hermes,nothink
HuggingFaceTB/SmolLM3-3B=smollm3_3b=load+long+gen=parser=hermes,nothink
ibm-granite/granite-3.3-2b-instruct=granite33_2b=load+long+gen=parser=granite
```

`nothink` sends `chat_template_kwargs {"enable_thinking": false}` with every request, so Qwen3 and SmolLM3 answer directly like the others. `mm=` turns off the image, audio and video inputs of multimodal checkpoints. A unit test checks every line against the lineup and against the tool-call parsers registered in the pinned vLLM build.

---

#### Step 3 — Launch One Chip per Tier

`tpu/launch.sh` bundles the committed tree, uploads it, and creates a flex-start queued resource whose startup script serves every arm in turn, uploads results after each one, and deletes the TPU at the end:

```bash
tpu/launch.sh arms-v5e1-tier2.txt --go
```

```text
arms:    tpu/arms-v5e1-tier2.txt (7 arms)
bundle:  gs://aisprint-491218-bucket/model-sweep/inputs/sweep-86d97ba02-0837b648.tgz
results: gs://aisprint-491218-bucket/model-sweep/2026-10-12-v5e1-tier2/
Created queued_resource [ms-2026-10-12-v5e1-tier2-1349].
created ms-2026-10-12-v5e1-tier2-1349; it deletes itself when the last arm finishes, or at 4h
```

Both tiers ran on `vllm/vllm-tpu@sha256:19a1a0526476f902eb83e1057f3d8938f35b457dd7716a30e9ab4f7bee90d507`, vLLM `0.29.1rc1.dev468+g0b7f11a1e`, so the tier 2 models pair with tier 1's Gemma run:

```text
[sweep 2026-10-11T17:14:28Z] start 2026-10-11-v5e1-tier1 on us-west4-a, arms arms-v5e1-tier1.txt
[sweep 2026-10-11T18:24:30Z] DONE: all arms attempted
[sweep 2026-10-12T13:51:26Z] start 2026-10-12-v5e1-tier2 on us-west4-a, arms arms-v5e1-tier2.txt
[sweep 2026-10-12T16:02:12Z] DONE: all arms attempted
```

---

#### Step 4 — Measure Speed and Accuracy the Same Way

Every model is served with `--max-model-len 4096 --max-num-seqs 16` and prefix caching on. Each one gets three measurements:

- **Output speed:** 256 output tokens per request at 1, 4 and 16 requests at once, median of three passes
- **Long prompts:** the same with 1,024- and 3,072-token prompts that never repeat, so the prefix cache cannot help
- **Accuracy:** all 1,319 GSM8K problems, greedy, with a 2,048-token answer limit, and the 400 BFCL simple tool-call records for models with a tool-call parser

Answers are scored in code. Differences against Gemma are paired on the same problems, with a 95% range from 10,000 resamples; a range that includes zero means no measured difference.

---

#### Step 5 — File the Results and Build the Report

```bash
python3 sweep_report.py file 2026-10-12-v5e1-tier2
python3 sweep_report.py report 2026-10-12-v5e1-tier2 --baseline-run 2026-10-11-v5e1-tier1
```

```text
filed smollm3_3b -> tpu-vllm-v5e1-smollm3_3b/benchmarks/runs/2026-10-12-v5e1-tier2
wrote results/2026-10-12-v5e1-tier2/REPORT.md (6 served, 1 failed)
```

`file` moves each model's logs and scored answers into its own directory. `report` refuses to pair two runs that served different image digests.

---

#### How Accurate Is Each Model?

Against Gemma 4 E2B, in percentage points, with the 95% range:

| Model | GSM8K | vs Gemma | BFCL | vs Gemma |
|---|---|---|---|---|
| Gemma 4 E2B | 0.911 | — | 0.927 | — |
| Qwen3-4B | 0.915 | +0.4 (-1.2 to +2.0) | 0.940 | +1.2 (-1.8 to +4.2) |
| Phi-4-mini | 0.895 | -1.6 (-3.2 to -0.1) | not scored | — |
| SmolLM3-3B | 0.858 | -5.3 (-7.2 to -3.5) | 0.945 | +1.7 (-1.0 to +4.8) |
| Qwen3-1.7B | 0.827 | -8.4 (-10.5 to -6.4) | 0.935 | +0.8 (-2.2 to +3.8) |
| Granite 3.3 2B | 0.817 | -9.5 (-11.6 to -7.4) | 0.920 | -0.7 (-3.8 to +2.2) |
| Qwen2.5-1.5B | 0.729 | -18.2 (-20.6 to -15.8) | 0.910 | -1.7 (-5.2 to +1.8) |
| Qwen3-0.6B | 0.599 | -31.2 (-34.0 to -28.5) | 0.825 | -10.3 (-14.2 to -6.5) |
| Qwen2.5-0.5B | 0.469 | -44.3 (-47.0 to -41.4) | 0.818 | -11.0 (-14.8 to -7.2) |
| SmolLM2-1.7B | 0.461 | -45.0 (-47.8 to -42.2) | no parser | — |

Five models are level with Gemma on BFCL tool calls, down to Qwen2.5-1.5B. Only Qwen3-4B is level on GSM8K, and from 1.7B down, Qwen3-1.7B aside, the score falls by 18 to 45 points.

---

#### How Fast Is Each Model?

Output tokens per second with 16 requests at once, and against Gemma:

| Model | Weights GiB | 16 requests | vs Gemma | 16 × 3,072-token prompts | vs Gemma |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.92 | 4176 | 2.21x | 2596 | 2.48x |
| Qwen3-0.6B | 1.40 | 3210 | 1.70x | 1301 | 1.24x |
| Qwen2.5-1.5B | 2.88 | 2780 | 1.47x | 1570 | 1.50x |
| Qwen3-1.7B | 3.78 | 2049 | 1.09x | 926 | 0.88x |
| Gemma 4 E2B | 8.97 | 1887 | 1.00x | 1049 | 1.00x |
| Granite 3.3 2B | 4.92 | 1761 | 0.93x | 732 | 0.70x |
| SmolLM3-3B | 6.22 | 1440 | 0.76x | 725 | 0.69x |
| Phi-4-mini | 8.31 | 1136 | 0.60x | 404 | 0.39x |
| Qwen3-4B | 7.49 | 1080 | 0.57x | 373 | 0.36x |

Gemma carries the largest weights in the table and still out-serves every model with more than 4 GiB of weights. With 3,072-token prompts the gap widens: Granite and SmolLM3 drop to about 0.7x, and Qwen3-4B to 0.36x.

---

#### Why Does Gemma Hold Up on Long Prompts?

The KV cache. Each model's boot log reports how many tokens of cache fit in the HBM left after its weights:

| Model | KV heads | Head size | Layers | KV cache tokens |
|---|---|---|---|---|
| Gemma 4 E2B | 1 | 256 / 512 | 35, 20 sharing | 321,280 |
| Granite 3.3 2B | 8 | 64 | 40 | 125,184 |
| SmolLM3-3B | 4 | 128 | 36 | 120,320 |
| Qwen3-1.7B | 8 | 128 | 28 | 100,096 |
| Qwen3-4B | 8 | 128 | 36 | 50,688 |

Gemma stores one KV head per layer, 28 of its 35 layers keep only a 512-token sliding window, and 20 layers reuse an earlier layer's cache. That leaves it 2.6 to 6.3 times the cache of the four models below it, even though its weights are the largest.

The same features that make Gemma awkward to port are the ones paying for its long-prompt speed on this chip.

---

#### 🔎 Tip: Turn Thinking Off Per Request

Qwen3 and SmolLM3 think by default, which spends the GSM8K answer budget on reasoning text. Passing the chat template switch with each request turns it off without a different checkpoint:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

With it, Qwen3-4B leaves no GSM8K answer at the 2,048-token limit, and SmolLM3-3B leaves 16 of 1,319.

---

#### 🔎 Tip: DeepSeek-R1-Distill-Qwen-1.5B Needs a One-Line Fix on TPU

The MIT-licensed R1 distill stops at load on this image:

```text
FAILED deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (exited) after 75s: AttributeError: 'Qwen2Config' object has no attribute 'text_config'. Did you mean: 'get_text_config'?
```

`tpu_inference/models/jax/qwen2.py` builds an untied `lm_head` with `getattr(hf_config, 'hidden_size', hf_config.text_config.hidden_size)`. Python evaluates the default argument first, so any Qwen2 checkpoint with `tie_word_embeddings: false` fails. The Qwen2.5 Instruct checkpoints tie their embeddings and load.

---

#### Compare and Contrast

| Model | Size | GSM8K | BFCL | 16 requests | Long prompts |
|---|---|---|---|---|---|
| Gemma 4 E2B | 5.12B | 0.911 | 0.927 | 1.00x | 1.00x |
| Qwen3-4B | 4.02B | 0.915 | 0.940 | 0.57x | 0.36x |
| SmolLM3-3B | 3.08B | 0.858 | 0.945 | 0.76x | 0.69x |
| Qwen3-1.7B | 2.03B | 0.827 | 0.935 | 1.09x | 0.88x |
| Granite 3.3 2B | 2.53B | 0.817 | 0.920 | 0.93x | 0.70x |

Long prompts are 16 requests of 3,072 tokens; every ratio is against Gemma on the same image.

---

#### So, Which One?

For the best accuracy per chip, Gemma 4 E2B: no other model matches both its scores and its speed on a v5e-1. For accuracy with standard attention, Qwen3-4B, level with Gemma on both tests at a bit over half the output. For tool calling at Gemma's size with the plainest architecture, SmolLM3-3B: level on BFCL, 5.3 points lower on math, three quarters of the speed. For tool calling at close to Gemma's speed, Granite 3.3 2B or Qwen3-1.7B, if math matters less. The models from 1.7B down run faster than Gemma but, apart from Qwen3-1.7B, give up 18 to 45 points of GSM8K.

---

#### Teardown

Each queued resource deletes itself after its last model. Confirm nothing is left:

```bash
gcloud alpha compute tpus queued-resources list --zone us-west4-a
```

---

#### Summary

The goal of this article was to find which small Apache-2.0 or MIT model can stand in for Gemma 4 E2B on one TPU v5e chip. The key to the solution was serving every model on one pinned vLLM image with identical settings, scoring every answer in code, and pairing each model against Gemma on the same problems. The results were:

- 🟢 Qwen3-4B is level with Gemma on GSM8K and BFCL, at 0.57x the output
- 🟢 SmolLM3-3B, Granite 3.3 2B, Qwen3-1.7B and Qwen2.5-1.5B are level with Gemma on BFCL tool calls
- 🟢 Gemma has the largest weights in the lineup and still out-serves every model with more than 4 GiB of weights
- ⚠️ Gemma's shared, single-head, sliding-window KV cache holds 2.6 to 6.3 times the tokens of Granite, SmolLM3, Qwen3-1.7B and Qwen3-4B, and keeps its lead with 3,072-token prompts
- ⚠️ Every model under 2B parameters except Qwen3-1.7B scores 18 to 45 points below Gemma on GSM8K
- ❌ DeepSeek-R1-Distill-Qwen-1.5B and OLMo 2 1B do not load on this image

Scope: one TPU v5e chip (`v5litepod-1`) in `us-west4-a`, flex-start, two runs on 2026-10-11 and 2026-10-12 on the same image digest, vLLM `0.29.1rc1.dev468+g0b7f11a1e`. Speed is the median of three passes; accuracy is one greedy pass. Gemma's figures come from the first run and pair with the second through the shared image. OLMo 2 1B fails to compile (it asks for 133.56 MB of the chip's 128 MB vector memory). Ministral 3 3B and Qwen3.5 0.8B and 2B served but are left out of the tables: on this image Ministral returned server errors on part of each test and Qwen3.5 returned unusable text, so their scores describe the serving path. Phi-4-mini produced no parsed tool calls with the `phi4_mini_json` parser and is reported on GSM8K and speed only. Parts of the analysis and writing were done with AI assistance (Claude); every figure comes from the committed report files.

The strategy for surveying small open models against Gemma 4 on a TPU v5e was validated with an incremental step by step approach.

---

#### References

- Code, logs, reports and scored answers: https://github.com/xbill9/model-sweep
- vLLM TPU: https://github.com/vllm-project/tpu-inference
- Gemma 4 E2B: https://huggingface.co/google/gemma-4-E2B-it
- Qwen3-4B: https://huggingface.co/Qwen/Qwen3-4B
- SmolLM3-3B: https://huggingface.co/HuggingFaceTB/SmolLM3-3B
- Granite 3.3 2B: https://huggingface.co/ibm-granite/granite-3.3-2b-instruct
- GSM8K: https://huggingface.co/datasets/openai/gsm8k
- Berkeley Function Calling Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
- Cloud TPU v5e: https://cloud.google.com/tpu/docs/v5e
