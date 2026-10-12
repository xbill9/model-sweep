#!/usr/bin/env python3
"""File a finished sweep run into its rigs, and build the comparison table from what was filed.

    python3 sweep_report.py file <run> [--prefix tpu-vllm-v5e1]   # results/<run>/ -> <rig>/benchmarks/runs/<run>/
    python3 sweep_report.py report <run> [--baseline gemma4_2b] [--baseline-run <run>]   # writes results/<run>/REPORT.md

`file` moves each arm's logs and records into the rig named for its chip and checkpoint (NAMING.md, "Harness
and rigs"). Run-wide files (run.log, vllm-version.txt, the checkpoint-named boot logs) stay in results/<run>/.
`report` reads only files; every ratio and difference in REPORT.md is computed here. --baseline-run takes the
baseline arm from another run, which must have served the same image digest (checked from run.log).
"""
import argparse
import glob
import json
import os
import random
import re
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))


def arm_tags(logs):
    return sorted({m.group(1) for f in os.listdir(logs) if (m := re.match(r"([a-z0-9]+_[a-z0-9]+)\.", f))})


def file_run(run, prefix):
    src = os.path.join(ROOT, "results", run)
    logs = os.path.join(src, "logs")
    for tag in arm_tags(logs):
        dst = os.path.join(ROOT, f"{prefix}-{tag}", "benchmarks", "runs", run)
        os.makedirs(os.path.join(dst, "logs"), exist_ok=True)
        for f in glob.glob(os.path.join(logs, f"{tag}.*")):
            shutil.move(f, os.path.join(dst, "logs", os.path.basename(f)))
        rec = os.path.join(src, "results", f"{run}-{tag}-gen")
        if os.path.isdir(rec):
            shutil.move(rec, os.path.join(dst, "gen"))
        print(f"filed {tag} -> {os.path.relpath(dst, ROOT)}")


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def boot_facts(path):
    """Weights resident and KV cache tokens, from vLLM's own log lines."""
    out = {}
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return out
    if m := re.search(r"total_hbm_used_gb=([\d.]+)GiB", text):
        out["weights_gib"] = float(m.group(1))
    if m := re.search(r"KV cache size: ([\d,]+) tokens", text):
        out["kv_tokens"] = int(m.group(1).replace(",", ""))
    return out


def collect(run):
    runlog = open(os.path.join(ROOT, "results", run, "logs", "run.log"), errors="replace").read()
    arms = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "*", "benchmarks", "runs", run))):
        rig = d.split(os.sep)[-4]
        tag = rig.split("-", 3)[3]
        L = os.path.join(d, "logs")
        meta = load(os.path.join(L, f"{tag}.meta.json")) or {}
        a = dict(rig=rig, tag=tag, model=meta.get("model"), sha=(meta.get("sha") or "")[:9], **boot_facts(os.path.join(L, f"{tag}.boot.log")))
        if m := re.search(rf"READY {re.escape(meta.get('model') or '?')} after (\d+)s", runlog):
            a["boot_s"] = int(m.group(1))
        if m := re.search(rf"FAILED {re.escape(meta.get('model') or '?')} \((\w+)\)[^:]*: (.*)", runlog):
            a["failed"] = m.group(2).strip()
        for c in (1, 4, 16):
            if d_ := load(os.path.join(L, f"{tag}.load.c{c}.json")):
                a[f"load{c}"] = d_["output_tok_per_s"]
            for n in (1024, 3072):
                if d_ := load(os.path.join(L, f"{tag}.long{n}.c{c}.json")):
                    a[f"long{n}_{c}"] = d_["output_tok_per_s"]
                    a[f"ttft{n}_{c}"] = d_["ttft_median_s"]
        for task in ("gsm8k", "bfcl_simple"):
            if g := load(os.path.join(L, f"{tag}.gen-{task}.json")):
                a[task] = g
        arms[tag] = a
    return arms


def records(rig, run, task):
    path = os.path.join(ROOT, rig, "benchmarks", "runs", run, "gen", f"{task}.jsonl")
    try:
        return {r["id"]: bool(r["right"]) for r in map(json.loads, open(path))}
    except OSError:
        return None


def paired_range(a, b, n=10000):
    """95% bootstrap range, in points, of accuracy(a) - accuracy(b) over the records both answered."""
    if not a or not b:
        return None
    d = [a[i] - b[i] for i in sorted(a.keys() & b.keys())]
    rng = random.Random(0)
    s = sorted(sum(rng.choice(d) for _ in d) / len(d) * 100 for _ in range(n))
    return s[int(n * 0.025)], s[int(n * 0.975)]


def fmt(v, spec):
    return "" if v is None else format(v, spec)


def image_digest(run):
    m = re.search(r"image \S*@(sha256:\w+)", open(os.path.join(ROOT, "results", run, "logs", "run.log"), errors="replace").read())
    return m and m.group(1)


def report(run, baseline, baseline_run=None):
    arms = collect(run)
    base_run = baseline_run or run
    if base_run != run and image_digest(base_run) != image_digest(run):
        raise SystemExit(f"{base_run} and {run} served different images; their numbers do not pair")
    base = (arms if base_run == run else collect(base_run)).get(baseline, {})
    if not base:
        raise SystemExit(f"no {baseline} arm in {base_run}")
    served = [a for a in arms.values() if "failed" not in a]
    served.sort(key=lambda a: -(a.get("load16") or 0))
    version = open(os.path.join(ROOT, "results", run, "logs", "vllm-version.txt")).read().strip().splitlines()[-1]
    lines = [
        f"# {run}",
        "",
        f"Generated by `sweep_report.py report {run}` from the filed logs. One TPU v5e chip, {version}, every arm at "
        "`--max-model-len 4096 --max-num-seqs 16`. Throughput is output tokens per second with 256 output tokens "
        "per request (`ignore_eos`), median of three passes. Ratios are against "
        f"`{baseline}`" + (f" from run `{base_run}` on the same image" if base_run != run else "") + ".",
        "",
        "#### Throughput",
        "",
        "| Rig | Weights GiB | KV tokens | Boot s | 1 req | 4 req | 16 req | 16 req vs base | 16 × 3,072-token prompts | TTFT s |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    rows = served if base_run == run else [base] + served
    for a in rows:
        ratio = a["load16"] / base["load16"] if a.get("load16") and base.get("load16") else None
        lines.append(
            f"| `{a['rig']}` | {fmt(a.get('weights_gib'), '.2f')} | {fmt(a.get('kv_tokens'), ',')} | {fmt(a.get('boot_s'), 'd')} | "
            f"{fmt(a.get('load1'), '.0f')} | {fmt(a.get('load4'), '.0f')} | {fmt(a.get('load16'), '.0f')} | "
            f"{fmt(ratio, '.2f')}x | {fmt(a.get('long3072_16'), '.0f')} | {fmt(a.get('ttft3072_16'), '.2f')} |")
    lines += [
        "",
        "#### Accuracy",
        "",
        "GSM8K: 1,319 problems, greedy, 2,048-token answer limit. BFCL simple: 400 records, only for arms with a "
        "tool-call parser. Points are percentage points against the baseline, paired on the same records, with a "
        "95% bootstrap range (10,000 resamples). A range that includes zero means no measured difference.",
        "",
        "| Rig | GSM8K | right / n | truncated | vs base (95% range) | BFCL | right / n | vs base (95% range) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    bg, bb = base.get("gsm8k", {}).get("accuracy"), base.get("bfcl_simple", {}).get("accuracy")
    for a in sorted(rows, key=lambda a: -(a.get("gsm8k", {}).get("accuracy") or 0)):
        g, b = a.get("gsm8k", {}), a.get("bfcl_simple", {})
        dg = (g["accuracy"] - bg) * 100 if g.get("accuracy") is not None and bg is not None else None
        db = (b["accuracy"] - bb) * 100 if b.get("accuracy") is not None and bb is not None else None
        rg = rb = ""
        if a is not base:
            if r := paired_range(records(a["rig"], run, "gsm8k"), records(base.get("rig", ""), base_run, "gsm8k")):
                rg = f" ({r[0]:+.1f} to {r[1]:+.1f})"
            if r := paired_range(records(a["rig"], run, "bfcl_simple"), records(base.get("rig", ""), base_run, "bfcl_simple")):
                rb = f" ({r[0]:+.1f} to {r[1]:+.1f})"
        lines.append(
            f"| `{a['rig']}` | {fmt(g.get('accuracy'), '.3f')} | {g.get('right', '')} / {g.get('n', '')} | "
            f"{g.get('truncated', '')} | {fmt(dg, '+.1f')}{rg} | {fmt(b.get('accuracy'), '.3f')} | "
            f"{b.get('right', '') if b else ''}{' / ' + str(b['n']) if b else ''} | {fmt(db, '+.1f')}{rb} |")
    failed = [a for a in arms.values() if "failed" in a]
    if failed:
        lines += ["", "#### Did not serve", ""]
        lines += [f"- `{a['rig']}` (`{a['model']}`): {a['failed']}" for a in failed]
    lines += ["", "#### Checkpoints", ""]
    lines += [f"- `{a['rig']}`: `{a['model']}` at `{a['sha']}`" for a in sorted(arms.values(), key=lambda a: a["rig"])]
    out = os.path.join(ROOT, "results", run, "REPORT.md")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {os.path.relpath(out, ROOT)} ({len(served)} served, {len(failed)} failed)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("file")
    f.add_argument("run")
    f.add_argument("--prefix", default="tpu-vllm-v5e1")
    r = sub.add_parser("report")
    r.add_argument("run")
    r.add_argument("--baseline", default="gemma4_2b")
    r.add_argument("--baseline-run")
    a = ap.parse_args()
    file_run(a.run, a.prefix) if a.cmd == "file" else report(a.run, a.baseline, a.baseline_run)


if __name__ == "__main__":
    main()
