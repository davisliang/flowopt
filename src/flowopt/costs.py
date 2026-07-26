"""Estimating what a search will cost, before committing to it.

A search spends money in four places, and they are not equally predictable:

1. **Analysis** — a handful of structured calls, plus dataset generation when the
   task brings no data. Small and steady.
2. **The design agent** — an Opus session per round. This is usually the largest
   single item and the least predictable: it depends on how much the agent reads,
   searches and re-tests. It bills through the SDK, not through our meter, so it
   is only knowable from what past runs actually recorded.
3. **Scoring candidates on dev** — rounds × candidates × dev examples × the cost
   of one query. That last term varies by two orders of magnitude between a
   single cheap call and a five-sample vote, which is most of the uncertainty.
4. **Scoring the frontier on test**, plus a judge call per graded answer when the
   task is judged.

So the estimate is a range, not a number, and every figure it rests on is
reported alongside it. Where past runs on this machine measured something, that
measurement is used and labelled `measured`; otherwise a documented default is
used and labelled `default`. An estimate that hides which is which invites more
trust than it has earned.
"""
import json
from dataclasses import dataclass, field
from statistics import median

from .models import ModelCatalog
from .runtime import ANSWER_SCHEMA

# Defaults used only until this machine has measured the real thing. Each is a
# rough central value; the range below widens generously around them.
DEFAULT_AGENT_COST_PER_ROUND = 1.20   # USD, an Opus design session
DEFAULT_CANDIDATES_PER_ROUND = 4.5    # the skill asks for 4-5
DEFAULT_COST_PER_QUERY = 0.0015       # a mid-range workflow: a call or two
DEFAULT_ANALYSIS_COST = 0.05          # analyzer + case types
COST_PER_GENERATED_EXAMPLE = 0.004    # Opus writing labelled examples
JUDGE_COST_PER_GRADE = 0.0004         # one cheap judged answer

# How far the range spreads around the central estimate. Workflow cost per query
# is the dominant unknown — a vote-of-five costs ~10x a single call — so the
# spread is wide on purpose. Narrow bounds here would be false precision.
LOW_FACTOR, HIGH_FACTOR = 0.45, 2.6


@dataclass
class Estimate:
    """What a search is expected to cost, and what that rests on.

    Attributes:
        low: Low end of the plausible range, USD.
        expected: Central estimate, USD.
        high: High end, USD.
        breakdown: Named parts of the central estimate, USD each.
        assumptions: Human-readable statements of every figure used, each marked
            "measured" (from past runs here) or "default".
        based_on_runs: How many past runs the measured figures came from.
    """
    low: float = 0.0
    expected: float = 0.0
    high: float = 0.0
    breakdown: dict = field(default_factory=dict)
    assumptions: list = field(default_factory=list)
    based_on_runs: int = 0


def observed(runs: list) -> dict:
    """Measure what past runs on this machine actually cost, per task.

    Cost per query is a property of the TASK, not of the machine: one measured
    ARC workflow cost $1.52 a query while an ifeval one cost $0.0013 — a factor
    of a thousand. Pooling those into a single median makes both estimates
    meaningless, so figures are kept per task and only fall back to the pool
    when a task has no history of its own.

    Args:
        runs: `(task, events)` pairs, events as `runstore.read_events` returns.

    Returns:
        `{"tasks": {task: figures}, "pooled": figures, "n_runs": n}`. Each
        `figures` holds whichever of `agent_cost_per_round`,
        `candidates_per_round`, `cost_per_query`, `cost_per_query_low` and
        `cost_per_query_high` the runs evidence.
    """
    by_task: dict[str, dict] = {}
    contributing = 0

    for task, events in runs:
        bucket = by_task.setdefault(task, {"agent": [], "per_round": [], "query": []})
        rounds, candidates, saw = set(), 0, False
        for event in events:
            kind = event.get("event")
            if kind == "agent_cost":
                bucket["agent"].append(event["usd"])
                saw = True
            elif kind == "round_start":
                rounds.add(event["round"])
            elif kind == "candidate":
                candidates += 1
                bucket["query"].append(event["dev_cost"])
                saw = True
        if rounds and candidates:
            bucket["per_round"].append(candidates / len(rounds))
        contributing += 1 if saw else 0

    def figures(bucket: dict) -> dict:
        """Reduce one bucket of raw observations to the figures the estimator uses."""
        out = {}
        if bucket["agent"]:
            out["agent_cost_per_round"] = median(bucket["agent"])
        if bucket["per_round"]:
            out["candidates_per_round"] = median(bucket["per_round"])
        if bucket["query"]:
            out["cost_per_query"] = median(bucket["query"])
            # The spread within a task is the real uncertainty: the designer will
            # try both a single cheap call and a five-sample vote.
            out["cost_per_query_low"] = min(bucket["query"])
            out["cost_per_query_high"] = max(bucket["query"])
        return out

    pooled = {"agent": [], "per_round": [], "query": []}
    for bucket in by_task.values():
        for key in pooled:
            pooled[key].extend(bucket[key])

    return {"tasks": {task: figures(b) for task, b in by_task.items()},
            "pooled": figures(pooled), "n_runs": contributing}


def estimate(cfg, history: dict = None, generates_data: bool = None,
             judged: bool = None, probe: "Probe" = None,
             available: int = None, allocated: dict = None) -> Estimate:
    """Estimate the cost of running a search with this config.

    Args:
        cfg: The resolved config the search would use.
        history: `observed(...)` output. Anything it provides is preferred over
            the defaults, and labelled as measured.
        generates_data: Whether examples will be generated rather than loaded.
            Inferred from `cfg.task.dataset` when omitted.
        judged: Whether grading calls a model per answer. Inferred from
            `cfg.task.check_type` when omitted; unknowable before analysis for a
            free-text task, where it is assumed False and said so.
        probe: A `run_probe` measurement. When given it replaces both history and
            defaults for cost per query — it is the only input measured on THIS
            task, right now, and it needs no prior runs to exist.
        available: How many examples the task's dataset actually holds. The run
            scores `min(n_examples, available)`, so an estimate that ignores this
            can be out by the ratio between them.

    Returns:
        An Estimate. Its `assumptions` list is the point: every figure it used,
        and whether that figure was measured here or is a default.
    """
    history = history or {}
    notes, breakdown = [], {}
    task_name = cfg.task.name
    mine = (history.get("tasks") or {}).get(task_name, {})
    pooled = history.get("pooled") or {}

    def figure(key, fallback, describe, transfers=True):
        """Take the most specific measurement available, and say which it was.

        Args:
            key: Which figure to look up.
            fallback: The documented default.
            describe: Renders the chosen value as a human-readable clause.
            transfers: Whether a measurement from OTHER tasks is evidence for
                this one. True for properties of the search process — how many
                candidates a round produces, what a design session costs. False
                for cost per query, which is a property of the task: ARC measured
                $0.74 a query against ifeval's $0.0018, so borrowing across tasks
                is worse than the default, not better.
        """
        if key in mine:
            notes.append(f"{describe(mine[key])} — measured on {task_name}")
            return mine[key]
        if transfers and key in pooled:
            notes.append(f"{describe(pooled[key])} — measured on other tasks; "
                         f"nothing on {task_name} yet")
            return pooled[key]
        if key in pooled:
            notes.append(f"{describe(fallback)} — default. Other tasks have "
                         f"measurements, but cost per query is task-specific "
                         f"(ARC ran ~400x ifeval), so they are not used here")
        else:
            notes.append(f"{describe(fallback)} — default, nothing measured yet")
        return fallback

    rounds = int(cfg.designer.rounds)
    n_train = max(0, int(cfg.data.n_train))
    alloc_val = int((allocated or {}).get("val") or 0)
    alloc_test = int((allocated or {}).get("test") or 0)
    explicit = int(cfg.data.n_dev) > 0 and int(cfg.data.n_test) > 0
    if explicit:
        dev, test = int(cfg.data.n_dev), int(cfg.data.n_test)
        if alloc_val:
            dev = min(dev, alloc_val)      # a partition can't give more than it has
        if alloc_test:
            test = min(test, alloc_test)
        n_examples = n_train + dev + test
        notes.append(f"explicit split sizes: {n_train} train / {dev} dev / {test} test")
    elif alloc_val and alloc_test:
        # The run will draw dev/test from routerllm's allocated partitions, and
        # blank inputs take a partition WHOLE — bbeh's 417-example holdout would
        # otherwise cost ~5x an estimate that assumed the fraction split.
        dev = min(int(cfg.data.n_dev), alloc_val) if int(cfg.data.n_dev) > 0 else alloc_val
        test = min(int(cfg.data.n_test), alloc_test) if int(cfg.data.n_test) > 0 else alloc_test
        n_examples = n_train + dev + test
        notes.append(f"sizes from routerllm's allocation: {dev} dev (its val partition), "
                     f"{test} test (its holdout) — blank inputs take each partition whole")
    else:
        # What will actually be scored, which is not always what was asked for:
        # a supplied dataset may hold fewer examples than the setting requests.
        n_examples = int(cfg.data.n_examples)
        if available is not None:
            n_examples = min(n_examples, available) if n_examples > 0 else available
            if available < int(cfg.data.n_examples):
                notes.append(f"the dataset has {available} examples, fewer than the "
                             f"{int(cfg.data.n_examples)} requested")
        # train is carved out first; dev and test split the remainder
        pool = max(2, n_examples - n_train)
        dev = max(1, int(pool * float(cfg.data.dev_fraction)))
        test = max(1, pool - dev)

    if generates_data is None:
        generates_data = not cfg.task.dataset
    if judged is None:
        judged = (cfg.task.check_type or "") == "llm_judge"

    # 1. analysis, and generating examples when the task brings none
    breakdown["analysis"] = DEFAULT_ANALYSIS_COST
    notes.append(f"analysis ~${DEFAULT_ANALYSIS_COST:.2f} — default")
    if generates_data:
        breakdown["generate examples"] = n_examples * COST_PER_GENERATED_EXAMPLE
        notes.append(f"generating {n_examples} examples at "
                     f"~${COST_PER_GENERATED_EXAMPLE:.3f} each — default")
    else:
        notes.append("dataset supplied, so nothing is generated")

    # 2. the design agent — usually the largest item
    agent = figure("agent_cost_per_round", DEFAULT_AGENT_COST_PER_ROUND,
                   lambda v: f"design agent ~${v:.2f} per round × {rounds}")
    breakdown["design agent"] = agent * rounds

    # 3. scoring every candidate on dev
    candidates = figure("candidates_per_round", DEFAULT_CANDIDATES_PER_ROUND,
                        lambda v: f"{v:.1f} candidates per round")
    probe_range = None
    if probe is not None and probe.n:
        catalog = ModelCatalog.from_config(cfg)
        probe_low, per_query, probe_high, why = per_query_from_probe(catalog, probe)
        probe_range = (probe_low, probe_high)
        notes.append(f"probed {probe.n} example(s) on {probe.model}: "
                     f"{probe.input_tokens:.0f} input / {probe.output_tokens:.0f} output "
                     f"tokens per call, costing ${probe.cost:.4f} to measure")
        notes.append(why)
        notes.append(f"${per_query:.5f} per query for a typical workflow — derived from "
                     f"that probe, not from past runs")
    else:
        per_query = figure("cost_per_query", DEFAULT_COST_PER_QUERY,
                           lambda v: f"${v:.5f} per query for a typical workflow",
                           transfers=False)
    total_candidates = candidates * rounds
    breakdown["score on dev"] = total_candidates * dev * per_query
    notes.append(f"dev split {dev} of {n_examples} examples "
                 f"(dev_fraction {cfg.data.dev_fraction})")

    # the agent tests its own candidates against the train split while iterating
    breakdown["agent self-tests"] = total_candidates * n_train * per_query

    # 4. the frontier on test — only the non-dominated candidates get there
    finalists = max(1.0, min(total_candidates, 3.0))
    breakdown["score on test"] = finalists * test * per_query
    notes.append(f"assuming ~{finalists:.0f} finalists reach the held-out test split")

    # 5. judged grading is an extra call per graded answer
    if judged:
        graded = total_candidates * dev + finalists * test
        breakdown["llm judge"] = graded * JUDGE_COST_PER_GRADE
        notes.append(f"judged grading adds ~${JUDGE_COST_PER_GRADE:.4f} per answer")
    elif (cfg.task.check_type or "") == "" and not cfg.task.description:
        notes.append("grading rule not known until the task is analyzed; assuming "
                     "it is not model-judged, which would cost more if it is")

    expected = sum(breakdown.values())

    # Where this task's own spread is known, scale the range by it rather than by
    # a made-up factor: the cheapest and dearest workflow actually seen bound the
    # query-cost term far better than a guess does.
    scored_queries = total_candidates * dev + finalists * test
    if probe_range:
        fixed = expected - scored_queries * per_query
        low = fixed + scored_queries * probe_range[0]
        high = fixed + scored_queries * probe_range[1]
        notes.append(f"range spans the cheapest and dearest workflow the probe implies: "
                     f"${probe_range[0]:.5f} to ${probe_range[1]:.5f} per query")
    elif "cost_per_query_low" in mine and per_query > 0:
        fixed = expected - scored_queries * per_query
        low = fixed + scored_queries * mine["cost_per_query_low"]
        high = fixed + scored_queries * mine["cost_per_query_high"]
        notes.append(f"range spans the cheapest and dearest workflow measured on "
                     f"{task_name}: ${mine['cost_per_query_low']:.5f} to "
                     f"${mine['cost_per_query_high']:.5f} per query")
    else:
        low, high = expected * LOW_FACTOR, expected * HIGH_FACTOR
        notes.append(f"range is a flat {LOW_FACTOR}x-{HIGH_FACTOR}x — no measured "
                     f"spread for {task_name} to derive it from")

    return Estimate(low=max(0.0, low), expected=expected, high=high,
                    breakdown={k: round(v, 4) for k, v in breakdown.items()},
                    assumptions=notes, based_on_runs=history.get("n_runs", 0))


# A probe measures one cheap call on real examples. Turning that into a range for
# what the DESIGNER will build needs assumptions about shape, which are documented
# here rather than buried:
# Calibrated against two measured searches — ifeval (cheap regime) and ARC
# (escalated). Two anchors is thin, so these are multipliers on a MEASURED
# baseline rather than absolute guesses, which is what keeps the error bounded
# when a new task looks like neither.
#   ifeval: probe call $0.0022, search ran $0.0013-$0.0285, median $0.0018
#   ARC:    probe call $0.0171, search ran $0.0123-$1.5239, median $0.7372
CHEAP_CALLS_TYPICAL = 1.5     # the cheap model works: a call, sometimes a check
CHEAP_CALLS_HIGH = 5.0        # ...or a five-sample vote on the middle model
ESCALATED_CALLS_TYPICAL = 4.0   # it doesn't: draft, verify, revise, on the big model
ESCALATED_CALLS_HIGH = 8.0
# Real workflows write shorter answers than the probe's schema-constrained one,
# so the cheapest candidate lands below a single probe call.
LOW_CALL_FRACTION = 0.6
EFFORT_OUTPUT_MULTIPLIER = 3.0   # thinking tokens count against output
# Below this, the cheap model cannot do the task, so the designer escalates and
# nearly every candidate ends up on an expensive model. Measured on ARC: Haiku
# scored 0.00 and the search ran ~60x the baseline cost. Above it, cheap
# workflows survive and cost stays near the baseline.
ESCALATION_ACCURACY = 0.35


@dataclass
class Probe:
    """One cheap call on real examples, measured.

    Attributes:
        input_tokens: Median input tokens for a single call on this task.
        output_tokens: Median output tokens.
        output_tokens_high: The largest output seen. On an open-ended task the
            same prompt can produce 89 tokens or 2118 depending on whether the
            model tries or gives up, and that spread drives the cost — so the
            high bound uses this rather than the median.
        accuracy: How the cheapest model scored answering directly. Low means the
            designer will escalate, which is what makes a task expensive.
        cost: What the probe itself actually spent, USD.
        n: How many examples were probed.
        model: The model the probe ran on.
    """
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    output_tokens_high: float = 0.0
    accuracy: float = 0.0
    cost: float = 0.0
    n: int = 0
    model: str = ""


def run_probe(cfg, client, grader, examples: list, n: int = 5) -> Probe:
    """Measure one direct call on this task, to anchor an estimate in fact.

    Costs a few cents and takes seconds. Nothing about the task is assumed: the
    prompts are real, so the token counts are this task's, and the answers are
    graded, so the accuracy says whether the cheap tier can do the work.

    Args:
        cfg: The run config.
        client: A ModelClient.
        grader: The task's grader, used to score the probe's answers.
        examples: Real examples to probe with.
        n: How many to run.

    Returns:
        A Probe. `n` is 0 if there were no examples to run.
    """
    chosen = examples[:n]
    if not chosen:
        return Probe()

    model = client.catalog.default
    inputs, outputs, scores, spent = [], [], [], 0.0
    for item in chosen:
        # A probe is a courtesy, not the job. An overloaded API or a refusal must
        # not take down the estimate — fewer samples, or none, is the right
        # failure, and the caller falls back to defaults when n is 0.
        try:
            response = client.call(model, item["question"], schema=ANSWER_SCHEMA)
        except Exception:
            continue
        usage = response.usage
        inputs.append(usage["input"] + usage["cache_read"] + usage["cache_write"])
        outputs.append(usage["output"])
        spent += client.catalog.cost_usd(model, usage)
        try:
            answer = json.loads(response.text).get("answer", response.text)
        except ValueError:
            answer = response.text
        try:
            scores.append(grader.score(str(answer), item))
        except Exception:
            scores.append(0.0)

    if not inputs:
        return Probe(model=model)
    return Probe(input_tokens=median(inputs), output_tokens=median(outputs),
                 output_tokens_high=max(outputs),
                 accuracy=sum(scores) / len(scores), cost=spent,
                 n=len(inputs), model=model)


def per_query_from_probe(catalog, probe: Probe) -> tuple[float, float, float, str]:
    """Price a typical designed workflow from a probe, for every model tier.

    The probe measures one call on the cheap model. What the search actually
    builds is some number of calls, possibly on a bigger model, possibly
    thinking. Those token counts are the task's; the price table does the rest,
    so no further API calls are needed to price the expensive tiers.

    Args:
        catalog: The ModelCatalog, for prices.
        probe: The measurement.

    Returns:
        `(low, expected, high, reasoning)` — dollars per query, plus a sentence
        explaining which regime was assumed and why.
    """
    def call_cost(model_id: str, thinking: bool = False, worst: bool = False) -> float:
        spec = catalog.spec(model_id)
        base = probe.output_tokens_high if worst else probe.output_tokens
        output = base * (EFFORT_OUTPUT_MULTIPLIER if thinking else 1.0)
        return (probe.input_tokens * spec.price_in + output * spec.price_out) / 1_000_000

    ids = catalog.ids
    cheapest, dearest = ids[0], ids[-1]
    middle = ids[len(ids) // 2]

    low = LOW_CALL_FRACTION * call_cost(cheapest)

    spread = (probe.output_tokens_high / probe.output_tokens) if probe.output_tokens else 1.0
    if probe.accuracy < ESCALATION_ACCURACY:
        expected = ESCALATED_CALLS_TYPICAL * call_cost(dearest, thinking=True)
        high = ESCALATED_CALLS_HIGH * call_cost(dearest, thinking=True, worst=True)
        why = (f"the cheap model scored {probe.accuracy:.2f} on the probe, below "
               f"{ESCALATION_ACCURACY} — it cannot do this task, so the designer will "
               f"escalate and most candidates will run on {dearest} with thinking on"
               + (f". Reply length varied {spread:.0f}x across the probe, so the upper "
                  f"bound is wide" if spread > 3 else ""))
    else:
        expected = CHEAP_CALLS_TYPICAL * call_cost(cheapest)
        high = CHEAP_CALLS_HIGH * call_cost(middle, worst=True)
        why = (f"the cheap model scored {probe.accuracy:.2f} on the probe, so cheap "
               f"workflows are viable and most candidates should stay on {cheapest}")
    return low, expected, high, why
