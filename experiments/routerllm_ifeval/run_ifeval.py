"""Optimize a workflow for routerllm's ifeval task and report it against that
repo's recorded baselines.

Everything task-agnostic comes from `flowopt`; everything specific to this
comparison is in `config/task/ifeval.yaml` (the task, its grader, a tighter
per-query budget) and in the reporting below.

  selection : 100 train examples (split 60 dev / 40 internal test)
  reporting : the router's 46-example holdout14 subset, untouched during selection

Usage: python -u run_ifeval.py [--rounds N] [--smoke]

Use `-u`: the design agent's subprocess output is block-buffered otherwise and
the run looks hung when it isn't.
"""
import argparse
import json
import pathlib
import time

from flowopt import Session, analysis
from flowopt.optimizer import optimize

HERE = pathlib.Path(__file__).parent

# routerllm's recorded numbers on the same 46 examples, for context in the output.
BASELINES = {"haiku": 0.848, "opus": 0.891, "router": 0.848, "oracle": 0.957}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--smoke", action="store_true", help="tiny run to prove the wiring")
    ap.add_argument("--out", default=str(HERE / "ifeval_result.json"))
    args = ap.parse_args()

    overrides = [f"designer.rounds={1 if args.smoke else args.rounds}"]
    if args.smoke:
        overrides.append("data.n_examples=8")
    session = Session.load("ifeval", overrides)
    cfg = session.cfg

    benchmark = analysis.build_benchmark(cfg, session.client)
    if args.smoke:                       # 8 examples is enough to prove the wiring
        benchmark.dev, benchmark.test = benchmark.dev[:5], benchmark.test[:3]
    held_out = [json.loads(line) for line in open(HERE / "ifeval_test.jsonl")]

    evaluator = session.evaluator(benchmark.grader)
    started = time.time()
    search = optimize(cfg, benchmark, evaluator, log=lambda *a: print(*a, flush=True))
    if not search.archive:
        raise SystemExit("no candidates survived — nothing to report")

    # Score EVERY candidate, not just the frontier: ties on a 60-example dev set
    # are common, and the frontier silently drops the tied-but-pricier ones.
    print(f"\nscoring all {len(search.archive)} candidates on internal + held-out test", flush=True)
    results = []
    for candidate in search.archive:
        internal = evaluator.run(candidate.program, benchmark.test)
        held = evaluator.run(candidate.program, held_out)
        results.append({
            "name": candidate.name, "description": candidate.description, "code": candidate.code,
            "dev_accuracy": candidate.dev.accuracy, "dev_cost": candidate.dev.cost,
            "internal_accuracy": internal.accuracy, "internal_cost": internal.cost,
            "test_accuracy": held.accuracy, "test_cost": held.cost, "test_n": len(held_out),
            "test_answers": [{"prompt_hash": item["prompt_hash"], "answer": record["answer"],
                              "score": record["score"], "error": record["error"]}
                             for item, record in zip(held_out, held.records, strict=True)],
        })
        print(f"  {candidate.name:26} dev {candidate.dev.accuracy:.3f} | "
              f"internal {internal.accuracy:.3f} | TEST {held.accuracy:.3f}  "
              f"${held.cost:.5f}/q", flush=True)

    spend = (sum(c.dev.cost * len(benchmark.dev) for c in search.archive)
             + sum(r["internal_cost"] * len(benchmark.test)
                   + r["test_cost"] * len(held_out) for r in results))
    pathlib.Path(args.out).write_text(json.dumps({
        "task": "ifeval", "rounds": cfg.designer.rounds,
        "n_dev": len(benchmark.dev), "n_internal": len(benchmark.test), "n_test": len(held_out),
        "baselines_test": BASELINES,
        "workflow_api_spend_usd": round(spend, 4),
        "wall_clock_s": round(time.time() - started, 1),
        "results": results,
    }, indent=1))
    print(f"\nworkflow API spend: ${spend:.2f}   wall clock: {(time.time()-started)/60:.1f} min")
    print(f"written: {args.out}")


if __name__ == "__main__":
    main()
