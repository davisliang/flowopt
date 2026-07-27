#!/usr/bin/env python3
"""Import MATH-500 from HuggingFace (HuggingFaceH4/MATH-500).

The 500-problem subset of the MATH benchmark that OpenAI selected for their
"Let's Verify Step by Step" work and that Artificial Analysis reports. Answers
are exact values, often LaTeX ("\\frac{1}{2}"), so grading uses
`check_type: llm_equality` — the same binary equality checker AA uses — rather
than numeric parsing.

Split into equal thirds (train/val/test) with a seeded shuffle, matching the
HLE import.

Usage:
    uv run --with pyarrow python scripts/import_math500.py
"""
import json
import pathlib
import random
import tempfile
import urllib.request

import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmarks" / "math_500"
SEED = 0

DESCRIPTION = ("Solve a MATH-500 competition mathematics problem. The answer is an "
               "exact value — an integer, fraction, or expression; an answer is "
               "correct only if it is mathematically equal to the reference.")


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def main() -> None:
    urls = json.loads(fetch(
        "https://huggingface.co/api/datasets/HuggingFaceH4/MATH-500/parquet/default/test"))
    rows = []
    for url in urls:
        with tempfile.NamedTemporaryFile(suffix=".parquet") as handle:
            handle.write(fetch(url))
            handle.flush()
            rows.extend(pq.read_table(handle.name,
                                      columns=["problem", "answer", "subject",
                                               "level"]).to_pylist())
    print(f"{len(rows)} problems")

    random.Random(SEED).shuffle(rows)
    third = len(rows) // 3
    examples = []
    for i, row in enumerate(rows):
        split = "train" if i < third else "val" if i < 2 * third else "test"
        examples.append({"question": row["problem"], "answer": str(row["answer"]),
                         "subject": row.get("subject"), "level": row.get("level"),
                         "split": split})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "data.jsonl", "w") as handle:
        for example in examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    counts = {s: sum(1 for e in examples if e["split"] == s)
              for s in ("train", "val", "test")}
    (OUT_DIR / "benchmark.yaml").write_text(f"""\
# math_500 — imported by scripts/import_math500.py from HuggingFaceH4/MATH-500.
# Graded with the binary equality checker (check_type: llm_equality), matching
# how Artificial Analysis scores it; accuracy is percentage correct.
name: math_500
description: >-
  {DESCRIPTION}
source_dataset: HuggingFaceH4/MATH-500
num_fewshot: 0
examples: {len(examples)}
sampled_from: {len(examples)}
grader_label: equality (official HLE checker)
grading_supported: true
aa_validated: true
# Split into equal thirds with a seeded shuffle (seed {SEED}); no external
# allocation or baselines apply to these splits.
split_labeled: {counts['train']} train / {counts['val']} val / {counts['test']} test
check_type: llm_equality
""")
    print(f"wrote {len(examples)} -> {OUT_DIR / 'data.jsonl'}  "
          f"({counts['train']} train / {counts['val']} val / {counts['test']} test)")


if __name__ == "__main__":
    main()
