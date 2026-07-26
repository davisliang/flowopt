"""Step 4 — present the tradeoffs and save the search.

Every finalist is a real option at a different accuracy/cost point; the frontier
is the set worth choosing between, and the two constrained picks answer the
questions people actually ask ("what can I afford?", "what's the cheapest thing
that's good enough?"). All numbers reported here are from the held-out test
split.
"""
import json
import pathlib

from .optimizer import TEST, Candidate, Search
from .pareto import best_under_budget, cheapest_above_accuracy


def summarize(search: Search, cfg, log=print) -> None:
    """Print the test-set frontier and the two constrained picks.

    Args:
        search: A completed Search. Its finalists must be test-scored.
        cfg: The run config, for `report.max_cost_per_query` and
            `report.min_accuracy`.
        log: Where the lines go.
    """
    if not search.finalists:
        log("No workflows survived — nothing to report.")
        return

    log("\nPareto frontier (cheapest -> most accurate):")
    for candidate in search.test_frontier():
        log(f"  {candidate.name:24s} accuracy {candidate.test.accuracy:.2f}   "
            f"${candidate.test.cost:.5f}/query")

    budget = cfg.report.max_cost_per_query
    best = best_under_budget(search.finalists, budget, on=TEST)
    log(f"\nBest workflow under ${budget}/query:  "
        + (f"{best.name}  (accuracy {best.test.accuracy:.2f}, ${best.test.cost:.5f})"
           if best else "none — every workflow costs more"))

    floor = cfg.report.min_accuracy
    cheapest = cheapest_above_accuracy(search.finalists, floor, on=TEST)
    log(f"Cheapest workflow at accuracy >= {floor}:  "
        + (f"{cheapest.name}  (accuracy {cheapest.test.accuracy:.2f}, "
           f"${cheapest.test.cost:.5f})"
           if cheapest else "none — no workflow reaches it"))


def print_code(search: Search, log=print) -> None:
    """Print each frontier workflow in full — this is what you would ship.

    Args:
        search: A completed Search.
        log: Where the lines go.
    """
    for candidate in search.test_frontier():
        log("=" * 72)
        log(f"{candidate.name}   —   accuracy {candidate.test.accuracy:.2f},  "
            f"${candidate.test.cost:.5f}/query")
        if candidate.description:
            log(candidate.description)
        log("-" * 72)
        log(candidate.code.strip() + "\n")


def save(search: Search, cfg, out_dir=None) -> pathlib.Path:
    """Write the search to JSON: every candidate, its scores, and its code.

    Per-example call traces are dropped — they hold live API objects and are for
    inspecting a search in memory, not for the record on disk.

    Args:
        search: A completed Search.
        cfg: The run config, for the task name and default output directory.
        out_dir: Directory to write to, overriding `cfg.report.output_dir`.

    Returns:
        Path to the written file, `<out_dir>/<task name>.json`.
    """
    out_dir = pathlib.Path(out_dir or cfg.report.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg.task.name}.json"
    path.write_text(json.dumps({
        "task": cfg.task.name,
        "description": cfg.task.description or cfg.task.seed_prompt,
        "rounds": cfg.designer.rounds,
        "candidates": [_as_dict(c) for c in search.archive],
        "frontier": [c.name for c in search.test_frontier()],
    }, indent=1))
    return path


def plot(search: Search, path=None, show=False):
    """Plot every finalist, with the frontier drawn through the non-dominated ones.

    Args:
        search: A completed Search.
        path: Where to save the figure. None skips saving.
        show: Whether to call `plt.show()` — for notebooks.

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    frontier = search.test_frontier()
    figure = plt.figure(figsize=(7, 5))
    for candidate in search.finalists:
        plt.scatter(candidate.test.cost, candidate.test.accuracy, color="#888888")
        plt.annotate(candidate.name, (candidate.test.cost, candidate.test.accuracy),
                     fontsize=8, xytext=(5, 4), textcoords="offset points")
    plt.plot([c.test.cost for c in frontier], [c.test.accuracy for c in frontier],
             "-o", color="#d9534f", label="Pareto frontier")
    plt.xlabel("cost per query (USD)")
    plt.ylabel("accuracy")
    plt.title("LLM-designed workflows: accuracy vs. cost")
    plt.legend()
    plt.tight_layout()
    if path:
        figure.savefig(path, dpi=150)
    if show:
        plt.show()
    return figure


def _as_dict(candidate: Candidate) -> dict:
    """Render one candidate for the saved JSON.

    Args:
        candidate: The candidate to serialize.

    Returns:
        Its name, description and code, plus a "dev" and (if scored) "test" block
        of accuracy, cost, cached input share, and up to three error messages.
    """
    scores = {}
    for split in ("dev", "test"):
        score = getattr(candidate, split)
        if score:
            scores[split] = {"accuracy": score.accuracy, "cost_per_query": score.cost,
                             "cached_input_frac": score.cached_input_frac,
                             "errors": score.errors[:3]}
    record = {"name": candidate.name, "description": candidate.description,
              "code": candidate.code}
    if candidate.helpers:                    # the operators the code calls — kept so it runs
        record["helpers"] = candidate.helpers
    return {**record, **scores}
