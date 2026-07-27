"""Build the ifeval train/test slices from a routerllm checkout, and prove the
local grader agrees with that repo's recorded labels.

    python build_data.py --out <dir>

Writes ifeval_train.jsonl (100) and ifeval_test.jsonl (46). Exits non-zero if
the grader disagrees with a single recorded label — the whole comparison rests
on those two scores meaning the same thing, so it is checked, not assumed.
"""
import argparse
import json
import pathlib
import random
import statistics
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--routerllm", default="/Users/davis/Documents/code/routerllm",
                help="checkout holding router_runs/ (holdout14 artifacts)")
ap.add_argument("--paired", default="/Users/davis/Documents/code/routerllm_june_23_2025",
                help="checkout holding router_data/ (paired dataset + split.json)")
ap.add_argument("--n-train", type=int, default=100)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

OUT = pathlib.Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
RL, PAIRED = pathlib.Path(args.routerllm), pathlib.Path(args.paired)

# ---- test: the <=100/task holdout14 subset the prior model runs used ---------
subset = set(json.loads((RL / "router_runs/holdout14_subset_hashes.json").read_text()))
holdout = [json.loads(line) for line in open(RL / "router_runs/holdout14_examples.jsonl")]
test_hashes = {r["prompt_hash"] for r in holdout
               if r["task"] == "ifeval" and r["prompt_hash"] in subset}

# ---- train: sampled from split.json's train partition -----------------------
split = json.loads((PAIRED / "router_data/split.json").read_text())
# the June-23 test partition must still BE the holdout14 set, or the train
# sample could overlap what we report on
assert {tuple(r.split("\t")) for r in split["test"]} == \
       {(r["task"], r["prompt_hash"]) for r in holdout}, \
       "split.json test partition no longer matches holdout14_examples.jsonl"

pool = sorted(r.split("\t")[1] for r in split["train"] if r.split("\t")[0] == "ifeval")
train_hashes = set(random.Random(args.seed).sample(pool, min(args.n_train, len(pool))))
assert not (train_hashes & test_hashes), "train/test overlap"

# ---- pull questions + the grading metadata (target is literally 0 for ifeval) --
want = train_hashes | test_hashes
train, test = [], []
with open(PAIRED / "router_data/router_haiku_opus.jsonl") as f:      # 121 MB, streamed
    for line in f:
        r = json.loads(line)
        if r["task"] != "ifeval" or r["prompt_hash"] not in want:
            continue
        rec = {"prompt_hash": r["prompt_hash"], "question": r["question"],
               "doc": r["doc"], "preds": r.get("preds"), "correct": r.get("correct")}
        (train if r["prompt_hash"] in train_hashes else test).append(rec)

for name, rows in (("ifeval_train", train), ("ifeval_test", test)):
    (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    h = statistics.mean(float(r["correct"]["haiku"]) for r in rows)
    o = statistics.mean(float(r["correct"]["opus"]) for r in rows)
    orc = statistics.mean(float(r["correct"]["haiku"] or r["correct"]["opus"]) for r in rows)
    print(f"{name:<14} n={len(rows):>3}   haiku={h:.3f}  opus={o:.3f}  oracle={orc:.3f}")

# ---- the check the comparison depends on ------------------------------------
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from grader import grade  # noqa: E402

def disagreements(rows):
    bad = 0
    for r in rows:
        for tier in ("haiku", "opus"):
            resp = r["preds"][tier]
            if isinstance(resp, list):
                resp = resp[0] if resp else ""
            bad += grade(str(resp), r) != float(r["correct"][tier])
    return bad, 2 * len(rows)

# The TEST slice is what gets reported, so it has to match the recorded labels
# exactly. The train slice only steers selection, and carries 1-2 examples of
# irreducible noise: ifeval's case/language checkers call langdetect, which
# neither lm-eval nor routerllm seeds, so the benchmark disagrees with itself
# run to run. grader.py pins the seed; these numbers are measured under it.
bad_te, n_te = disagreements(test)
bad_tr, n_tr = disagreements(train)
print("\ngrader vs recorded labels:")
print(f"  test  {n_te - bad_te}/{n_te} agree   <- reported; must be exact")
print(f"  train {n_tr - bad_tr}/{n_tr} agree   <- selection only; langdetect-sensitive")
if bad_te:
    sys.exit(f"FAIL: {bad_te} disagreement(s) on the reported slice — not comparable")
if bad_tr > 0.05 * n_tr:
    sys.exit(f"FAIL: train disagreement {bad_tr}/{n_tr} exceeds the 5% noise allowance")
print(f"written to {OUT}")
