#!/usr/bin/env python3
"""Rebuild a run's result.json from its event log.

A run stopped before the ranking step never wrote one, so the UI shows no chart
and no table even though the candidates were scored and paid for. Everything
needed is still in `events.jsonl`, except the programs' source — which only the
finished result carries — so recovered candidates have their scores but not
their code.

Runs from the current build save partial results on stop by themselves; this is
for runs that predate that.

    uv run python scripts/recover_run_result.py <run_id> [--write]
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from flowopt import runstore              # noqa: E402
from flowopt.pareto import pareto_front   # noqa: E402
from flowopt.runtime import SplitScore    # noqa: E402


def rebuild(run_id: str) -> dict:
    """Assemble a result payload from one run's events.

    Args:
        run_id: The run to rebuild.

    Returns:
        The same shape `report.save` writes: task, description, rounds,
        candidates, frontier. Candidates carry `"code": ""`.
    """
    status = runstore.read_status(run_id)
    if status is None:
        raise SystemExit(f"no such run: {run_id}")

    candidates: dict[str, dict] = {}
    for event in runstore.read_events(run_id):
        if event.get("event") == "candidate":
            candidates[event["name"]] = {
                "name": event["name"], "description": event.get("description", ""),
                "code": "",
                "dev": {"accuracy": event["dev_accuracy"],
                        "cost_per_query": event["dev_cost"],
                        "cached_input_frac": event.get("cached_input_frac", 0.0),
                        "errors": event.get("errors", [])}}
        elif event.get("event") == "test_scored" and event["name"] in candidates:
            candidates[event["name"]]["test"] = {"accuracy": event["test_accuracy"],
                                                 "cost_per_query": event["test_cost"]}

    scored = list(candidates.values())
    # No test scores means the run never ranked; the frontier is then the best
    # dev tradeoffs, which is what the search itself was steering by.
    split = "test" if any(c.get("test") for c in scored) else "dev"
    points = [SplitScore(c["name"], c[split]["accuracy"], c[split]["cost_per_query"])
              for c in scored if c.get(split)]
    return {"task": status.task, "description": "(recovered from the event log)",
            "rounds": status.rounds, "candidates": scored,
            "frontier": [p.name for p in pareto_front(points)],
            "recovered": True, "frontier_split": split}


def main() -> int:
    """Rebuild one run's result, printing it or writing it. Returns an exit code."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id")
    ap.add_argument("--write", action="store_true",
                    help="write result.json (refuses to overwrite an existing one)")
    args = ap.parse_args()

    payload = rebuild(args.run_id)
    print(f"{len(payload['candidates'])} candidate(s), "
          f"frontier on {payload['frontier_split']}: {', '.join(payload['frontier'])}")
    for candidate in payload["candidates"]:
        dev = candidate["dev"]
        print(f"  {candidate['name']:28s} acc {dev['accuracy']:.3f}  "
              f"${dev['cost_per_query']:.4f}/query")

    if not args.write:
        print("\n(dry run — pass --write to save it)")
        return 0

    target = runstore.run_dir(args.run_id) / "result.json"
    if target.exists():
        print(f"\n{target} already exists; refusing to overwrite", file=sys.stderr)
        return 1
    target.write_text(json.dumps(payload, indent=1))
    print(f"\nwritten: {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
