#!/usr/bin/env python3
"""Score a candidate `solve(question, call_model)` program on the train set.

Run from the agent's working directory, which holds `run_config.yaml`,
`task_spec.json` and `train_task.json` — the examples the design agent may
see, disjoint from the dev split the search scores. This is a thin wrapper: the runtime,
metering and grading are the SAME code the final search uses
(`flowopt.runtime`), so a dev number here means what it means there.

    python eval_candidate.py <candidate.py>
"""
import json
import os
import sys

if os.environ.get("FLOWOPT_SRC"):    # set when the search spawns the agent
    sys.path.insert(0, os.environ["FLOWOPT_SRC"])

from flowopt.config import load_resolved    # noqa: E402
from flowopt.grading import Grader          # noqa: E402
from flowopt.session import Session         # noqa: E402


def main() -> None:
    """Score the candidate named on the command line, printing one JSON line.

    The contract with the agent is exactly one JSON object on stdout, always —
    so every failure, including a typo'd filename or a missing config, comes back
    as `{"ok": false, "error": ...}` rather than a traceback it has to interpret.
    """
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "error": "usage: eval_candidate.py <file.py>"}))
        return
    try:
        print(json.dumps(evaluate(sys.argv[1])))
    except Exception as error:
        print(json.dumps({"ok": False, "error": f"{type(error).__name__}: {error}"}))


def evaluate(candidate_path: str) -> dict:
    """Run one candidate against the staged train examples.

    Args:
        candidate_path: Path to a `.py` defining `solve(question, call_model)`.

    Returns:
        On success, `{"ok": True, "accuracy", "cost_per_query", "n",
        "cached_input_frac", "errors"}`. If the program didn't compile,
        `{"ok": False, "error": ...}`.
    """
    session = Session.from_config(load_resolved("run_config.yaml"))
    spec = json.load(open("task_spec.json"))
    train = json.load(open("train_task.json"))

    check = spec["check"]
    grader = (Grader.from_grader(spec["grader"]) if check["type"] == "custom"
              else Grader(kind=check["type"], client=session.client,
                          judge_model=session.cfg.judge.model,
                          task=check["task"], rubric=check["rubric"]))

    # Inject the run's operators (working_skills/helpers.py) exactly as the search
    # does, so a workflow that calls them scores here what it will score there.
    helpers = open("working_skills/helpers.py").read() if os.path.exists(
        "working_skills/helpers.py") else ""
    program = {"name": os.path.basename(candidate_path),
               "code": open(candidate_path).read(), "helpers": helpers}
    score = session.evaluator(grader).run(program, train)

    if score.error:
        return {"ok": False, "error": score.error}
    return {"ok": True, "accuracy": score.accuracy, "cost_per_query": score.cost,
            "n": len(train), "cached_input_frac": score.cached_input_frac,
            "errors": score.errors[:3]}


main()
