"""Import FanOutQA into the harness as a runnable task.

FanOutQA (Zhu et al., ACL 2024) is a fan-out multi-hop benchmark: each question
asks for a fact about EACH of many entities (typically 5+), so answering it well
needs decomposition and aggregation, not one shot. That makes it a probe for
whether the optimizer's search discovers multi-call structure — the answer is
world knowledge (code can't compute it) and the fan-out defeats a single call.

Downloads the dev split (which ships gold answers; the test split's are held out
on the leaderboard) from the source repo and writes:
  benchmarks/fanoutqa/data.jsonl     {"question", "answer", ...} per line
  benchmarks/fanoutqa/benchmark.yaml the descriptor the UI's task picker reads
  config/task/fanoutqa.yaml          the runnable task

    uv run python scripts/import_fanoutqa.py

Open-book on purpose (runtime.tools includes web_search): the workflow retrieves,
so decompose -> search -> aggregate is expressible. NOTE the gold answers are
pinned to a Nov-2023 Wikipedia snapshot, so volatile figures (populations, GDP)
have drifted; the loose-accuracy metric gives partial credit and both single-call
and multi-call workflows face the same drift, so the STRUCTURE gap stays valid
even though absolute accuracy is depressed.
"""
import argparse
import json
import pathlib
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEV_URL = "https://raw.githubusercontent.com/zhudotexe/fanoutqa/main/fanoutqa/data/fanout-final-dev.json"

DESCRIPTION = (
    "A fan-out multi-hop question: it asks for a specific fact about each of many "
    "entities (typically five or more), so answering it requires gathering facts "
    "about every entity and combining them. Answer with a single string that "
    "states the value for EVERY entity the question asks about (for example, a "
    "list of 'Entity: value' pairs). Grading is a loose string match in [0, 1]: it "
    "checks what fraction of the required entities and values from the reference "
    "answer appear in your answer, so a partial answer earns partial credit — "
    "cover all of them.")

# Real answers from the set, shown to the design agent purely as FORMAT examples.
ANSWER_EXAMPLES = [
    "Pat Burrell: Right, Mark Mulder: Left, Corey Patterson: Left, Jeff Austin: Right, JD Drew: Left",
    "Macau: 680000, Maldives: 521000, Singapore: 5917000, Bahrain: 1463000, Hong Kong: 7498000",
]

# The judge-graded variant grades completeness instead of substring presence, so
# its description leans on "cover all of them" rather than the string metric.
JUDGE_DESCRIPTION = (
    "A fan-out multi-hop question: it asks for a specific fact about each of many "
    "entities (typically five or more), so answering it requires gathering facts "
    "about every entity and combining them. Answer with a single string that states "
    "the value for EVERY entity the question asks about (for example, a list of "
    "'Entity: value' pairs). Cover all of them — completeness is graded.")

JUDGE_RUBRIC = (
    "The reference answer gives the correct value for each entity the question asks "
    "about (a set of entity -> value facts). Score how completely and correctly the "
    "candidate covers those facts, 0 to 100, in proportion to the fraction of the "
    "reference's entities for which the candidate gives the correct value: all "
    "correct = 100, half correct = about 50, none = 0. Judge factual content, not "
    "wording — ignore formatting, order, phrasing, and units, and count a numeric "
    "value as correct if it matches the reference within minor rounding (e.g. 1.03 "
    "billion vs 1.027 billion). A missing entity, a wrong value, or a fabricated "
    "entity earns no credit for that entity; do not reward fluent but incomplete "
    "answers.")


def main() -> None:
    """Download FanOutQA dev and write the data, descriptor, and task config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0, help="keep at most this many (0 = all)")
    args = parser.parse_args()

    with urllib.request.urlopen(DEV_URL) as response:
        records = json.loads(response.read().decode("utf-8"))
    if args.limit:
        records = records[: args.limit]

    rows = [{"question": r["question"], "answer": r["answer"],
             "id": r["id"], "categories": r.get("categories", [])} for r in records]

    bench_dir = ROOT / "benchmarks" / "fanoutqa"
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (bench_dir / "benchmark.yaml").write_text(
        "# fanoutqa — imported by scripts/import_fanoutqa.py from zhudotexe/fanoutqa (dev split)\n"
        "name: fanoutqa\n"
        f"description: >-\n  {DESCRIPTION}\n"
        "source_dataset: fanoutqa-dev\n"
        f"examples: {len(rows)}\n"
        "grading_supported: true\n"
        "check_type: exact\n"
        "routerllm_grader: loose-string\n"
        "grader: benchmarks/_graders/fanoutqa.py\n")

    examples_yaml = "\n".join(f"    - {json.dumps(e)}" for e in ANSWER_EXAMPLES)
    (ROOT / "config" / "task" / "fanoutqa.yaml").write_text(
        "# FanOutQA — a fan-out multi-hop task, to test whether the search finds\n"
        "# decompose -> retrieve -> aggregate structure. Open-book (web tools on).\n"
        "task:\n"
        "  name: fanoutqa\n"
        f"  description: >-\n    {DESCRIPTION}\n"
        "  check_type: exact\n"
        "  dataset: benchmarks/fanoutqa/data.jsonl\n"
        "  grader: benchmarks/_graders/fanoutqa.py\n"
        "  answer_examples:\n"
        f"{examples_yaml}\n"
        "runtime:\n"
        "  tools: [web_search, web_fetch]   # open-book: retrieve + aggregate\n"
        "data:\n"
        "  n_examples: 40     # 24 dev / 16 test at dev_fraction 0.6; raise for a fuller run\n")

    # --- judge-graded variant: same data, reference-based LLM judge -----------
    # Its own benchmark.yaml is what makes it a first-class entry in the UI picker
    # (with an example count and description), not just a bare config/task file.
    judge_dir = ROOT / "benchmarks" / "fanoutqa_judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    (judge_dir / "benchmark.yaml").write_text(
        "# fanoutqa_judge — the fanoutqa data, graded by a reference-based LLM judge\n"
        "name: fanoutqa_judge\n"
        f"description: >-\n  {JUDGE_DESCRIPTION}\n"
        "source_dataset: fanoutqa-dev\n"
        f"examples: {len(rows)}\n"
        "grading_supported: true\n"
        "check_type: llm_judge\n"
        "routerllm_grader: llm-judge\n")
    (ROOT / "config" / "task" / "fanoutqa_judge.yaml").write_text(
        "# FanOutQA graded by a reference-based LLM judge (see fanoutqa for the string\n"
        "# metric). Same data. Judge calls are evaluator cost, NOT workflow cost, are\n"
        "# non-deterministic, and are not comparable to the published string baselines.\n"
        "task:\n"
        "  name: fanoutqa_judge\n"
        f"  description: >-\n    {JUDGE_DESCRIPTION}\n"
        "  check_type: llm_judge\n"
        "  dataset: benchmarks/fanoutqa/data.jsonl\n"
        "  answer_examples:\n"
        f"{examples_yaml}\n"
        f"  judge_rubric: >-\n    {JUDGE_RUBRIC}\n"
        "runtime:\n"
        "  tools: [web_search, web_fetch]   # open-book: retrieve + aggregate\n"
        "data:\n"
        "  n_examples: 40     # 24 dev / 16 test at dev_fraction 0.6\n"
        "judge:\n"
        "  model: claude-sonnet-5   # a stronger judge than the haiku default for factuality\n")

    print(f"wrote {len(rows)} instances to {bench_dir/'data.jsonl'}")
    print("wrote benchmark.yaml + config/task/*.yaml for: fanoutqa, fanoutqa_judge")


if __name__ == "__main__":
    main()
