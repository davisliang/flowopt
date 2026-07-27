#!/usr/bin/env python3
"""Re-partition a benchmark's fixed split labels in place.

For benchmarks whose data is already complete locally but carries a partition
that no longer fits (gpqa_diamond_gen inherited routerllm's 154/19/25, capping
dev at 19), this rewrites the `split` field with a seeded shuffle: `--train`
rows to train, `--dev` rows to val, and the rest to test. Run-time sizing still
decides how much of each partition a search draws; these labels set the
ceilings. Omit the sizes for equal thirds, the full-set imports' scheme.

Any recorded baselines are dropped from benchmark.yaml: they were measured on
the old partition and don't transfer to the new splits.

Usage:
    uv run python scripts/resplit_benchmark.py gpqa_diamond_gen --train 64 --dev 32
    uv run python scripts/resplit_benchmark.py some_benchmark          # thirds
"""
import argparse
import json
import pathlib
import random

from omegaconf import OmegaConf

ROOT = pathlib.Path(__file__).resolve().parent.parent


def resplit(name: str, train: int, dev: int, seed: int) -> None:
    """Relabel one benchmark's rows and refresh its benchmark.yaml.

    Args:
        name: Directory under `benchmarks/`.
        train: Rows labeled train. 0 means a third.
        dev: Rows labeled val. 0 means a third.
        seed: Shuffle seed, recorded in the yaml.
    """
    folder = ROOT / "benchmarks" / name
    meta = OmegaConf.to_container(OmegaConf.load(folder / "benchmark.yaml"))
    rows = [json.loads(line) for line in open(folder / "data.jsonl")]

    train = train or len(rows) // 3
    dev = dev or len(rows) // 3
    if train + dev >= len(rows):
        raise SystemExit(f"{name}: {train} train + {dev} dev leaves no test "
                         f"rows out of {len(rows)}")

    random.Random(seed).shuffle(rows)
    for i, row in enumerate(rows):
        row["split"] = "train" if i < train else "val" if i < train + dev else "test"

    with open(folder / "data.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_test = len(rows) - train - dev
    label = meta.get("grader_label", meta.get("routerllm_grader", ""))
    aa = "aa_validated: true\n" if meta.get("aa_validated") else ""
    (folder / "benchmark.yaml").write_text(f"""\
# {name} — full local set, re-partitioned by scripts/resplit_benchmark.py
# (seed {seed}). Earlier baselines, if any, were measured on the previous
# partition and are dropped rather than misreported against these splits.
name: {name}
description: >-
  {str(meta.get('description', '')).strip()}
source_dataset: {meta.get('source_dataset', '')}
num_fewshot: {meta.get('num_fewshot', 0)}
examples: {len(rows)}
sampled_from: {meta.get('sampled_from', len(rows))}
grader_label: {label}
grading_supported: true
# Size the run's train/dev/test draws from these partitions when starting a
# search; blank dev/test in the form take a partition whole.
split_labeled: {train} train / {dev} val / {n_test} test
check_type: {meta.get('check_type', '')}
{aa}""")
    print(f"{name}: {len(rows)} rows -> {train} train / {dev} val / {n_test} test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", help="directories under benchmarks/")
    parser.add_argument("--train", type=int, default=0, help="train rows (0 = a third)")
    parser.add_argument("--dev", type=int, default=0, help="dev/val rows (0 = a third)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    for name in args.names:
        resplit(name, args.train, args.dev, args.seed)


if __name__ == "__main__":
    main()
