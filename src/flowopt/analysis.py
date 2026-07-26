"""Step 1 — what is this task, and how should an answer be graded?

The analyzer's outputs are Pydantic models rather than hand-written JSON Schema
dicts: one definition both constrains the model's reply and types the object read
back, so the two can't drift. Getting the examples themselves is `dataset`.

NOTE: a Pydantic docstring becomes the JSON Schema "description" and is sent to
the model with the request. Those docstrings are prompt text as well as comments
— keep developer asides in `#` comments, which are not transmitted.
"""
import random
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

from . import dataset as datasets
from . import prompts
from .grading import Grader


class TaskAnalysis(BaseModel):
    """How to run and grade this task, inferred from the task description.

    Attributes:
        description: One paragraph describing the task. Briefs the design agent
            and grounds the judge.
        check_type: Which grading rule fits — "numeric", "exact" or "llm_judge".
        judge_rubric: Grading criteria for free-form tasks; "" for the others.
        answer_examples: Correctly formatted answers. Shown to the design agent
            as the target format, and used to calibrate the rubric.
    """
    model_config = ConfigDict(extra="forbid")   # -> additionalProperties: false

    description: str
    check_type: Literal["numeric", "exact", "llm_judge"]
    judge_rubric: str
    answer_examples: list[str]


@dataclass
class Benchmark:
    """A task made measurable: what it is, how to grade it, and the data.

    This is what the optimizer searches against.

    Attributes:
        analysis: The task description and inferred answer format.
        grader: Scores a returned answer against an example.
        dev: Examples candidates are scored on during the search; their
            failures are fed back to later design rounds.
        test: Held-out examples; only the final ranking touches them.
        train: The only examples the design agent may SEE — its self-test set
            and any few-shot material — disjoint from dev and test, so nothing
            the agent tunes against is ever scored.
        judge_status: Human-readable note on how the judge was set up — whether
            the task-specific rubric passed calibration or a generic judge is in
            use. "" when nothing is judged by a model.
    """
    analysis: TaskAnalysis
    grader: Grader
    dev: list[dict]
    test: list[dict]
    train: list[dict] = field(default_factory=list)
    judge_status: str = ""

    @property
    def description(self) -> str:
        """The task description, as given to the design agent and the judge."""
        return self.analysis.description


def split_examples(cfg, data: list[dict], log=print) -> tuple[list, list, list]:
    """Split the pool into train / dev / test — allocated, sized, and random.

    Train is carved out FIRST: it is the only slice the design agent may see
    (self-tests, few-shot material), so nothing the agent tunes against is ever
    scored. Dev guides the search; test is held out for the final ranking.

    Examples carrying `"split"` labels are an external allocation — routerllm's
    own 80/10/10 partition. Each of our splits then draws ONLY from its
    counterpart: train from their train, dev from their val, test from their
    test (the holdout its baselines were measured on), so no example ever
    crosses a partition boundary and test numbers are computed on the same
    examples as the baselines. With only test labels, train/dev draw from the
    unlabeled rest; with no labels the split is random. Either way the
    sampling is seeded, so two runs at the same sizes score the same examples.

    Sizes come from `cfg.data`: `n_train` always; `n_dev`/`n_test` when both
    are set, else `n_examples` capped and `dev_fraction` split.

    Args:
        cfg: The run config; reads `cfg.data`.
        data: The loaded or generated examples.
        log: Where the sampling decisions are noted.

    Returns:
        `(train, dev, test)` — disjoint; train may be empty, dev and test never.

    Raises:
        ValueError: Too few examples to give dev and test at least one each.
    """
    n_train = max(0, int(cfg.data.n_train))
    n_dev, n_test = int(cfg.data.n_dev), int(cfg.data.n_test)
    explicit = n_dev > 0 and n_test > 0
    rng = random.Random(0)

    allocated = [r for r in data if r.get("split") in ("test", "holdout")]
    allocated_train = [r for r in data if r.get("split") == "train"]
    allocated_val = [r for r in data if r.get("split") == "val"]
    pool = [r for r in data if r.get("split") not in ("test", "holdout", "train", "val")]

    def pick(part: list, n: int) -> list:
        """A seeded sample of `n` from one partition; n<=0 or n>=len takes all."""
        part = sorted(part, key=lambda r: r["question"])
        return part if n <= 0 or n >= len(part) else rng.sample(part, n)

    if allocated_train or allocated_val:
        # A full routerllm allocation: our test = their test, our dev samples
        # their val, our train samples their train — each partition drawn from
        # the partition routerllm assigned it, never across.
        test = pick(allocated, n_test)
        train_pool = allocated_train + pool
        if allocated_val:
            dev = pick(allocated_val, n_dev)
        else:                      # allocation without val rows in this export
            rng.shuffle(train_pool)
            cut = max(1, n_dev or int(len(train_pool) * cfg.data.dev_fraction))
            dev, train_pool = train_pool[:cut], train_pool[cut:]
        train = pick(train_pool, n_train) if n_train > 0 else []
        log(f"splits drawn from routerllm's allocation: "
            f"{len(train)} of {len(train_pool)} train, "
            f"{len(dev)} of {len(allocated_val) or len(dev)} val->dev, "
            f"{len(test)} of {len(allocated)} test")
    elif allocated:
        test = (sorted(allocated, key=lambda r: r["question"]) if n_test <= 0
                or n_test >= len(allocated)
                else rng.sample(allocated, n_test))
        log(f"test     = {len(test)} externally allocated holdout example(s)"
            + (f" (of {len(allocated)})" if len(test) < len(allocated) else ""))
        budget = (n_train + n_dev) if explicit else max(1, int(cfg.data.n_examples) - len(test))
        pool = datasets.take(pool, budget, log=log)
        rng.shuffle(pool)
        n_train = min(n_train, max(0, len(pool) - 1))     # dev keeps at least one
        train = pool[:n_train]
        dev = pool[n_train:n_train + n_dev] if explicit else pool[n_train:]
    else:
        wanted = (n_train + n_dev + n_test) if explicit else int(cfg.data.n_examples)
        pool = datasets.take(list(data), wanted, log=log)
        rng.shuffle(pool)
        n_train = min(n_train, max(0, len(pool) - 2))     # always leave dev + test
        train, rest = pool[:n_train], pool[n_train:]
        if explicit:
            dev, test = rest[:n_dev], rest[n_dev:n_dev + n_test]
        else:
            cut = max(1, min(len(rest) - 1, int(len(rest) * cfg.data.dev_fraction)))
            dev, test = rest[:cut], rest[cut:]

    if not dev or not test:
        raise ValueError(f"need at least 1 dev and 1 test example, got "
                         f"{len(dev)} dev / {len(test)} test from {len(data)} examples")
    return train, dev, test


def build_benchmark(cfg, client, log=print) -> Benchmark:
    """Analyze the task, build its grader, and load or generate its data.

    Args:
        cfg: The run config.
        client: ModelClient used to analyze the task and generate examples.
        log: Where progress lines go. Pass a no-op to silence them.

    Returns:
        A Benchmark with train, dev and test splits; dev and test non-empty.

    Raises:
        ValueError: Too few examples to split. Dev drives the search and test
            is the only honest number, so an empty split would surface as a
            meaningless accuracy of 0.00.
    """
    data = datasets.load_examples(cfg.task.dataset)
    analysis = analysis_from_config(cfg) or analyze_task(cfg, client, cfg.task.seed_prompt, data)

    if cfg.task.grader:
        grader, judge_status = Grader.from_grader(cfg.task.grader), "n/a (custom grader)"
    elif analysis.check_type == "llm_judge":
        rubric, judge_status = calibrate_rubric(cfg, client, analysis)
        grader = Grader(kind="llm_judge", client=client, judge_model=cfg.judge.model,
                        task=analysis.description, rubric=rubric)
    else:
        grader, judge_status = Grader(kind=analysis.check_type), "n/a (not an LLM judge)"

    if data is None:
        explicit = int(cfg.data.n_dev) > 0 and int(cfg.data.n_test) > 0
        wanted = (int(cfg.data.n_train) + int(cfg.data.n_dev) + int(cfg.data.n_test)
                  if explicit else int(cfg.data.n_examples))
        data = datasets.generate_examples(cfg, client, analysis, log=log, n_examples=wanted)
    if len(data) < 2:
        raise ValueError(f"need at least 2 examples to split dev/test, got {len(data)}")
    check_grader(grader, data[0])
    train, dev, test = split_examples(cfg, data, log=log)
    benchmark = Benchmark(analysis=analysis, grader=grader,
                          train=train, dev=dev, test=test, judge_status=judge_status)

    log(f"check    = {grader.kind}")
    if grader.kind == "llm_judge":
        log(f"judge    = {judge_status}")
    if analysis.answer_examples:
        log(f"answers  = {', '.join(repr(e) for e in analysis.answer_examples[:3])}")
    log(f"{len(train) + len(dev) + len(test)} examples  ->  {len(train)} train / "
        f"{len(dev)} dev / {len(test)} test")
    return benchmark


def benchmark_to_dict(benchmark: Benchmark) -> dict:
    """Render a Benchmark as plain data, so a later run can reuse it exactly.

    A continued search must score against the SAME dev/test splits or its
    numbers aren't comparable — and a generated dataset cannot be regenerated
    identically. Saving the benchmark whole (analysis, splits, and how the
    grader was set up) is what makes continuation sound.

    Args:
        benchmark: The benchmark to save.

    Returns:
        A JSON-serializable dict `benchmark_from_dict` can rebuild from.
    """
    return {"analysis": benchmark.analysis.model_dump(),
            "train": benchmark.train, "dev": benchmark.dev, "test": benchmark.test,
            "judge_status": benchmark.judge_status,
            "grader": {"kind": benchmark.grader.kind,
                       "task": benchmark.grader.task,
                       "rubric": benchmark.grader.rubric}}


def benchmark_from_dict(cfg, client, saved: dict) -> Benchmark:
    """Rebuild a saved Benchmark without any API call.

    The saved rubric is the post-calibration one, so nothing is re-inferred or
    re-calibrated — the grader means exactly what it meant in the source run.

    Args:
        cfg: The run config, for the judge model and a custom grader's path.
        client: ModelClient a judge grader will call through.
        saved: `benchmark_to_dict` output.

    Returns:
        The rebuilt Benchmark.
    """
    analysis = TaskAnalysis(**saved["analysis"])
    grader_info = saved.get("grader") or {}
    if cfg.task.grader:
        grader = Grader.from_grader(cfg.task.grader)
    elif grader_info.get("kind") == "llm_judge":
        grader = Grader(kind="llm_judge", client=client, judge_model=cfg.judge.model,
                        task=grader_info.get("task", analysis.description),
                        rubric=grader_info.get("rubric", ""))
    else:
        grader = Grader(kind=grader_info.get("kind", analysis.check_type))
    return Benchmark(analysis=analysis, grader=grader,
                     train=list(saved.get("train", [])),   # absent in pre-split saves
                     dev=list(saved["dev"]), test=list(saved["test"]),
                     judge_status=saved.get("judge_status", ""))


def check_grader(grader: Grader, example: dict) -> None:
    """Prove the grader can actually grade this data, before a search starts.

    A grader that raises is caught per example and scored 0.0 — which is
    indistinguishable, in the results, from a workflow that answered wrongly. A
    whole run of zeros is the symptom, and it costs a full search to discover.
    This turns that into an immediate error naming what is missing: it once cost
    9 candidates × 120 examples to learn that a dataset lacked the `doc` field
    its grader reads.

    Args:
        grader: The grader the run will use.
        example: One dataset example, used as the probe.

    Raises:
        ValueError: The grader raised on a well-formed answer, so it cannot grade
            this dataset. Judge graders are skipped — probing one costs an API
            call, and its failure mode is a score, not an exception.
    """
    if grader.kind == "llm_judge":
        return
    try:
        grader.score(str(example.get("answer", "")), example)
    except Exception as error:
        raise ValueError(
            f"the grader cannot read this dataset: {type(error).__name__}: {error}. "
            f"The examples have keys {sorted(example)} — the grader needs something "
            f"else. Fix the data or the grader before spending a search on it.") from error


def analysis_from_config(cfg) -> Optional[TaskAnalysis]:
    """Read the task's shape straight from config, skipping the analyzer.

    For a task whose shape is already known — a benchmark with its own metric
    doesn't need a model to guess at it.

    Args:
        cfg: The run config.

    Returns:
        A TaskAnalysis built from `cfg.task`, or None when `task.description` is
        unset, which is the normal case.
    """
    if not cfg.task.description:
        return None
    return TaskAnalysis(description=cfg.task.description,
                        check_type=cfg.task.check_type or "exact",
                        judge_rubric=cfg.task.judge_rubric or "",
                        answer_examples=list(cfg.task.answer_examples))


def analyze_task(cfg, client, seed_prompt: str, examples=None) -> TaskAnalysis:
    """Infer the task's description, grading rule, judge rubric and answer format.

    One structured model call.

    Args:
        cfg: The run config.
        client: ModelClient to call.
        seed_prompt: The task in plain English.
        examples: Up to 5 of the task's own labeled examples, shown to ground the
            inference. None when the task brought no data.

    Returns:
        A validated TaskAnalysis.
    """
    shown = ""
    if examples:
        shown = "\n\nExamples (input -> answer):\n" + "".join(
            f"- {item['question']} -> {item['answer']}\n" for item in examples[:5])
    prompt = prompts.render("analyze_task", seed_prompt=seed_prompt, examples=shown,
                            min_gold_score=int(cfg.judge.min_gold_score * 100),
                            max_empty_score=int(cfg.judge.max_empty_score * 100))
    return client.parse(cfg.analysis_model, prompt, TaskAnalysis)


def calibrate_rubric(cfg, client, analysis: TaskAnalysis) -> tuple[str, str]:
    """Check that the judge rubric actually discriminates, and drop it if not.

    A rubric is only usable if it scores an ideal answer high and an empty answer
    low. One that does neither produces numbers that look like measurements but
    are noise, so it is thrown away for the generic judge.

    Args:
        cfg: The run config, for the calibration thresholds.
        client: ModelClient the judge calls.
        analysis: The task analysis carrying the candidate rubric.

    Returns:
        `(rubric, status)`. `rubric` is the original if it passed, or "" to fall
        back to the generic judge. `status` explains which, for the run log.
    """
    golds = [e for e in analysis.answer_examples if str(e).strip()][:2]
    if not golds:
        return analysis.judge_rubric, "no golds to validate; using rubric as-is"

    judge = Grader(kind="llm_judge", client=client, judge_model=cfg.judge.model,
                   task=analysis.description, rubric=analysis.judge_rubric)
    high = sum(judge.judge(gold, gold) for gold in golds) / len(golds)   # gold vs itself -> high
    low = sum(judge.judge("", gold) for gold in golds) / len(golds)      # empty answer   -> low

    if high >= cfg.judge.min_gold_score and low <= cfg.judge.max_empty_score:
        return analysis.judge_rubric, "ok"
    return "", f"rubric didn't discriminate (gold={high:.2f}, empty={low:.2f}); using generic judge"
