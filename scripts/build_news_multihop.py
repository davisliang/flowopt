"""Build the `news_multihop` benchmark from frozen, sourced, timestamped facts.

This task closes BOTH bypasses at once. Its questions are about events from the
week it was built (post-cutoff, so no recall — the knowledge bypass is shut), AND
each answer requires combining facts from 2-4 SEPARATE sources that no single
article aggregates (so one search/one call can't shortcut it — the single-call
bypass is shut). If elaborate structure ever beats a single call within this
harness, this is where it should show.

Source of truth: the newest `benchmarks/news_multihop*/sources.json` — each item
carries the composed question, the final answer, and the atomic `hops` (each with
its own source URL and snippet) plus a derivation, all stamped with the gather
date. Arithmetic is re-verified at build time from the hop-free `answer` field
only in spirit; the derivations were checked by hand and four high-profile deaths
were spot-checked against Wikipedia. Answers are resultative, so they do not drift.

The benchmark's NAME carries the freeze date (`news_multihop_<as_of>`), so the
name itself says when the data goes stale and should be recycled. Refreshing
sources.json with a new `as_of` and re-running writes a NEW stamped benchmark
beside the old one rather than silently replacing it.

    uv run python scripts/build_news_multihop.py
"""
import json
import pathlib

from build_news_current import newest_sources

ROOT = pathlib.Path(__file__).resolve().parent.parent

# The description the design agent sees (config/task/*.yaml -> the design prompt).
# Deliberately NEUTRAL: only the answer format. It must not reveal that the events
# are recent, that retrieval is needed, or that the answer is composed from several
# sources — the search discovering all of that on its own is the experiment. The
# questions themselves make clear when combination is needed (they ask for a sum,
# etc.), so the description does not need to say so.
TASK_DESCRIPTION = ("Answer the question with a single short answer — a number or a "
                    "name — and nothing else.")

# Human-only blurb for the UI benchmark picker; never reaches the model.
BENCH_DESCRIPTION = ("Cross-source multi-hop QA over recent (post-cutoff) news. The "
                     "model-facing task description is kept neutral so the search must "
                     "discover both that answers require web retrieval and that they "
                     "must be composed from several sources.")

ANSWER_EXAMPLES = ["188", "Wally Funk", "$27.83 billion", "Goldman Sachs"]


def main() -> None:
    """Write data.jsonl, the benchmark descriptor, and the task config."""
    src_path = newest_sources("news_multihop*")
    src = json.loads(src_path.read_text())
    as_of, window, items = src["as_of"], src["window"], src["items"]
    name = f"news_multihop_{as_of.replace('-', '')}"   # the name says when it goes stale

    rows = []
    for item in items:
        golds = [item["answer"], *[a for a in item.get("aliases", []) if a]]
        # Only question + answer reach the model. Provenance (as_of, answer_type,
        # categories, per-hop source URLs) stays in sources.json, so the sampled
        # examples the design agent is shown reveal neither recency nor the sources.
        rows.append({"question": item["question"], "answer": golds})

    bench_dir = ROOT / "benchmarks" / name
    bench_dir.mkdir(parents=True, exist_ok=True)
    if src_path.parent != bench_dir:       # a refreshed freeze lands in its own dir
        (bench_dir / "sources.json").write_text(src_path.read_text())
    (bench_dir / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (bench_dir / "benchmark.yaml").write_text(
        f"# {name} — cross-source multi-hop QA over recent news\n"
        f"# Timestamped: answers verified as of {as_of} (events from {window}).\n"
        f"name: {name}\n"
        f"description: >-\n  {BENCH_DESCRIPTION}\n"
        "source_dataset: news\n"
        f"as_of: {as_of}\n"
        f"window: {window!r}\n"
        f"examples: {len(rows)}\n"
        "grading_supported: true\n"
        "check_type: exact\n"
        "routerllm_grader: contains\n"
        "grader: benchmarks/_graders/contains.py\n")

    examples_yaml = "\n".join(f"    - {json.dumps(e)}" for e in ANSWER_EXAMPLES)
    (ROOT / "config" / "task" / f"{name}.yaml").write_text(
        "# Cross-source multi-hop QA over recent (post-cutoff) news. Closes both\n"
        "# bypasses: must retrieve (no recall) AND must combine facts from several\n"
        "# sources (no single-call shortcut). Open-book; code execution allowed.\n"
        "# NOTE (human context only; these comments never reach the model): the task\n"
        "# description below is kept NEUTRAL on purpose — it must not tell the design\n"
        "# agent that the answers are recent, that web search is needed, or that the\n"
        "# answer is composed from several sources, so that the search discovering all\n"
        "# of that on its own is a real result rather than a hint.\n"
        f"# Timestamped: answers as of {as_of} (events from {window}).\n"
        "task:\n"
        f"  name: {name}\n"
        f"  description: >-\n    {TASK_DESCRIPTION}\n"
        "  check_type: exact\n"
        f"  dataset: benchmarks/{name}/data.jsonl\n"
        "  grader: benchmarks/_graders/contains.py\n"
        "  answer_examples:\n"
        f"{examples_yaml}\n"
        "runtime:\n"
        "  tools: [web_search, web_fetch]   # retrieve per hop, then aggregate\n"
        "data:\n"
        f"  n_examples: {len(rows)}     # use all of them ({len(rows)} multi-hop questions)\n")

    hops = sum(len(item["hops"]) for item in items)
    print(f"wrote {len(rows)} multi-hop questions ({hops} sourced hops, as of {as_of})")
    print(f"wrote {bench_dir/'benchmark.yaml'} and config/task/{name}.yaml")


if __name__ == "__main__":
    main()
