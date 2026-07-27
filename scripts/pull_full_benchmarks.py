#!/usr/bin/env python3
"""Replace subsampled benchmark data with the full canonical sets.

For every benchmark whose routerllm import was a sample, pull the complete
source dataset from HuggingFace, rewrite `data.jsonl` in the exact row format
the existing grader reads, and relabel splits as seeded equal thirds
(train/val/test) — run-time sizing decides how much of each partition a search
actually uses. routerllm's partitions and baselines are dropped: they were
measured on the old samples and don't transfer.

Covered: mmlu_pro (12,032), bbeh_gen (4,520), minerva_math (MATH test, boxed
answers), aa_omniscience (600 — the full public slice; AA holds the rest back
against contamination), simpleqa (1,000 verified), nq_open_gen (3,610),
gsm_plus_mini (2,400 testmini — also fixes the answer field, which previously
held the whole chain-of-thought), ifeval (541).

Usage:
    uv run --with pyarrow python scripts/pull_full_benchmarks.py [names...]
"""
import json
import os
import pathlib
import random
import sys
import tempfile
import urllib.parse
import urllib.request

import pyarrow.parquet as pq
from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEED = 0
LETTERS = "ABCDEFGHIJ"


def fetch(url: str) -> bytes:
    headers = {}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as r:
        return r.read()


def parquet_rows(dataset: str, config: str, split: str, columns=None) -> list[dict]:
    """Every row of one dataset config/split, via the HF parquet API."""
    listing = f"https://huggingface.co/api/datasets/{dataset}/parquet/{config}/{split}"
    rows = []
    for url in json.loads(fetch(listing)):
        with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
            handle.write(fetch(url))
            handle.flush()
            rows.extend(pq.read_table(handle.name, columns=columns).to_pylist())
    return rows


def parquet_configs(dataset: str) -> list[str]:
    listing = json.loads(fetch(f"https://huggingface.co/api/datasets/{dataset}/parquet"))
    return sorted(listing)


def boxed_answer(solution: str):
    """The contents of the last \\boxed{...} in a MATH solution, brace-matched."""
    start = solution.rfind("\\boxed{")
    if start == -1:
        return None
    i, depth = start + len("\\boxed{"), 1
    for j in range(i, len(solution)):
        depth += {"{": 1, "}": -1}.get(solution[j], 0)
        if depth == 0:
            return solution[i:j]
    return None


# ---- one adapter per benchmark: source -> rows in the existing data format --

def mmlu_pro():
    rows = parquet_rows("TIGER-Lab/MMLU-Pro", "default", "test",
                        ["question", "options", "answer", "category"])
    return [{"question": "Question:\n" + r["question"] + "\nOptions:\n"
                         + "\n".join(f"{LETTERS[i]}. {opt}"
                                     for i, opt in enumerate(r["options"])),
             "answer": r["answer"], "category": r["category"]} for r in rows]


def bbeh_gen():
    out = []
    for config in parquet_configs("jgyasu/bbeh"):
        for r in parquet_rows("jgyasu/bbeh", config, "train", ["task", "input", "target"]):
            out.append({"question": r["input"], "answer": r["target"], "task": r["task"]})
    return out


def minerva_math():
    out, skipped = [], 0
    for config in parquet_configs("EleutherAI/hendrycks_math"):
        for r in parquet_rows("EleutherAI/hendrycks_math", config, "test",
                              ["problem", "level", "type", "solution"]):
            answer = boxed_answer(r["solution"])
            if answer is None:
                skipped += 1
                continue
            out.append({"question": "Problem:\n" + r["problem"] + "\n\nSolution:",
                        "answer": answer, "level": r["level"], "type": r["type"]})
    if skipped:
        print(f"  minerva_math: skipped {skipped} problems with no \\boxed answer")
    return out


def aa_omniscience():
    rows = parquet_rows("ArtificialAnalysis/AA-Omniscience-Public", "default", "train",
                        ["domain", "topic", "question", "answer"])
    return [{"question": r["question"], "answer": r["answer"],
             "domain": r["domain"], "topic": r["topic"]} for r in rows]


def simpleqa():
    rows = parquet_rows("google/simpleqa-verified", "simpleqa_verified", "eval",
                        ["problem", "answer", "topic"])
    return [{"question": r["problem"], "answer": r["answer"], "topic": r["topic"]}
            for r in rows]


NQ_PREFIX = ("Answer the question with a short answer only — just the answer itself, "
             "no explanation or punctuation.\n\nQuestion: ")


def nq_open_gen():
    rows = parquet_rows("google-research-datasets/nq_open", "nq_open", "validation")
    # answer is a JSON-encoded alias list — exactly what the contains grader parses
    return [{"question": NQ_PREFIX + r["question"] + "\nShort answer:",
             "answer": json.dumps(list(r["answer"]), ensure_ascii=False)} for r in rows]


def gsm_plus_mini():
    rows = parquet_rows("qintongli/GSM-Plus", "default", "testmini",
                        ["question", "answer", "perturbation_type"])
    return [{"question": "Question: " + r["question"] + "\nAnswer:",
             "answer": str(r["answer"]), "perturbation_type": r["perturbation_type"]}
            for r in rows]


def ifeval():
    rows = parquet_rows("google/IFEval", "default", "train")
    # parquet structs come back null-filled to a uniform schema, which is the
    # exact kwargs shape the lm-eval checker was validated against
    return [{"question": r["prompt"], "answer": "0",
             "doc": {"key": r["key"], "prompt": r["prompt"],
                     "instruction_id_list": r["instruction_id_list"],
                     "kwargs": r["kwargs"]}} for r in rows]


PULLS = {
    "mmlu_pro": mmlu_pro,
    "bbeh_gen": bbeh_gen,
    "minerva_math": minerva_math,
    "aa_omniscience": aa_omniscience,
    "simpleqa": simpleqa,
    "nq_open_gen": nq_open_gen,
    "gsm_plus_mini": gsm_plus_mini,
    "ifeval": ifeval,
}


def write_benchmark(name: str, examples: list[dict]) -> None:
    """Write data.jsonl (thirds split) and refresh benchmark.yaml counts."""
    folder = ROOT / "benchmarks" / name
    meta = OmegaConf.to_container(OmegaConf.load(folder / "benchmark.yaml"))

    random.Random(SEED).shuffle(examples)
    third = len(examples) // 3
    for i, example in enumerate(examples):
        example["split"] = "train" if i < third else "val" if i < 2 * third else "test"

    with open(folder / "data.jsonl", "w") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    counts = {s: sum(1 for e in examples if e["split"] == s)
              for s in ("train", "val", "test")}
    description = str(meta.get("description", "")).strip()
    (folder / "benchmark.yaml").write_text(f"""\
# {name} — full set, imported by scripts/pull_full_benchmarks.py.
# Replaces the routerllm sample; routerllm's partition and baselines are gone
# (they were measured on that sample and don't transfer to these splits).
name: {name}
description: >-
  {description}
source_dataset: {meta.get('source_dataset', '')}
num_fewshot: 0
examples: {len(examples)}
sampled_from: {len(examples)}
grader_label: {meta.get('grader_label', meta.get('routerllm_grader', ''))}
grading_supported: true
# Split into equal thirds with a seeded shuffle (seed {SEED}); size the run's
# train/dev/test draws from these partitions when starting a search.
split_labeled: {counts['train']} train / {counts['val']} val / {counts['test']} test
check_type: {meta.get('check_type', '')}
""")
    print(f"{name}: {len(examples)} examples "
          f"({counts['train']}/{counts['val']}/{counts['test']} train/val/test)")


def main() -> None:
    names = sys.argv[1:] or list(PULLS)
    unknown = set(names) - set(PULLS)
    if unknown:
        sys.exit(f"no puller for: {', '.join(sorted(unknown))} "
                 f"(have: {', '.join(sorted(PULLS))})")
    for name in names:
        print(f"pulling {name} …")
        write_benchmark(name, PULLS[name]())


if __name__ == "__main__":
    main()
