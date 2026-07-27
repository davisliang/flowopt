#!/usr/bin/env python3
"""Import the full text-only Humanity's Last Exam set from HuggingFace.

Replaces the 200-example routerllm sample with every text-only question
(the same subset Artificial Analysis evaluates on — multimodal questions are
excluded for comparability across models), split into equal thirds:

    train — the design agent may see and self-test on these
    val   — the optimizer's dev split (guides the search)
    test  — held out for the final ranking

Graded with `check_type: llm_equality` (the official HLE judge prompt, binary
correct/incorrect), so accuracy reads as percentage correct, the way AA and the
HLE paper report it.

Usage:
    HF_TOKEN=... uv run --with pyarrow python scripts/import_hle.py

cais/hle is gated: the HF account behind HF_TOKEN must have accepted its terms.
"""
import json
import os
import pathlib
import random
import sys
import tempfile
import urllib.request

import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmarks" / "hle"
SEED = 0

DESCRIPTION = ("Answer a Humanity's Last Exam question — expert-level, across many "
               "fields. Answers are short and specific; an answer is correct only if "
               "it matches the reference (binary equality check, no partial credit).")


def fetch(url: str, token: str) -> bytes:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request) as response:
        return response.read()


def main() -> None:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        sys.exit("set HF_TOKEN (cais/hle is gated)")

    urls = json.loads(fetch(
        "https://huggingface.co/api/datasets/cais/hle/parquet/default/test", token))
    rows = []
    for url in urls:
        with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
            handle.write(fetch(url, token))
            handle.flush()
            table = pq.read_table(handle.name,
                                  columns=["question", "image", "answer",
                                           "answer_type", "category"])
            rows.extend(table.to_pylist())

    text_only = [r for r in rows if not r.get("image")]
    print(f"{len(rows)} questions, {len(text_only)} text-only")

    random.Random(SEED).shuffle(text_only)
    third = len(text_only) // 3
    examples = []
    for i, row in enumerate(text_only):
        split = "train" if i < third else "val" if i < 2 * third else "test"
        examples.append({"question": row["question"], "answer": row["answer"],
                         "answer_type": row.get("answer_type"),
                         "category": row.get("category"), "split": split})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "data.jsonl", "w") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    counts = {s: sum(1 for e in examples if e["split"] == s)
              for s in ("train", "val", "test")}
    (OUT_DIR / "benchmark.yaml").write_text(f"""\
# hle — full text-only set, imported by scripts/import_hle.py from cais/hle.
# Same subset Artificial Analysis evaluates (multimodal questions excluded);
# graded with the official HLE equality-checker prompt (check_type:
# llm_equality), so accuracy is percentage correct as AA reports it.
name: hle
description: >-
  {DESCRIPTION}
source_dataset: cais/hle
num_fewshot: 0
examples: {len(examples)}
sampled_from: {len(examples)}
grader_label: equality (official HLE checker)
grading_supported: true
aa_validated: true
# Split into equal thirds with a seeded shuffle (seed {SEED}) — no routerllm
# allocation; the old 200-example routerllm sample and its baselines were
# replaced by this full import, so no external baselines apply to these splits.
split_labeled: {counts['train']} train / {counts['val']} val / {counts['test']} test
check_type: llm_equality
""")
    print(f"wrote {len(examples)} -> {OUT_DIR / 'data.jsonl'}  "
          f"({counts['train']} train / {counts['val']} val / {counts['test']} test)")


if __name__ == "__main__":
    main()
