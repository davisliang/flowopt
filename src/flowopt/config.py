"""The typed configuration schema, loaded from `config/` with OmegaConf.

One `Config` object is threaded through the whole run, so every knob has exactly
one home and appears in `config/config.yaml`. The dataclasses below are the
schema: OmegaConf validates the YAML against them, so a misspelled key or a
string where a number belongs fails at load time rather than mid-run.
"""
from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import OmegaConf

from .paths import CONFIG_DIR


@dataclass
class ModelSpec:
    """One model the optimizer may use, with the facts needed to price and route it.

    Attributes:
        id: The API model id, e.g. "claude-haiku-4-5".
        price_in: USD per 1,000,000 input tokens.
        price_out: USD per 1,000,000 output tokens.
        thinks: Whether the model supports the effort / adaptive-thinking
            parameters. A workflow asking for `effort` on a model that can't
            think is ignored rather than erroring.
    """
    id: str
    price_in: float
    price_out: float
    thinks: bool = False


@dataclass
class CallConfig:
    """Settings for a single model call.

    Attributes:
        max_output_tokens: Ceiling on one reply. NOT a cost knob — you are billed
            for the tokens a reply actually uses — so it is set high enough never
            to be the reason an answer is short. It must stay generous because
            thinking counts against it.
        max_tool_turns: How many times one call may be resumed while a
            server-side tool keeps pausing the turn.
        cache_write_multiplier: Multiple of the input rate billed for writing the
            prompt cache (a one-time surcharge).
        cache_read_multiplier: Multiple of the input rate billed for reading it
            (~90% discount).
    """
    max_output_tokens: int = 64000
    max_tool_turns: int = 5
    cache_write_multiplier: float = 1.25
    cache_read_multiplier: float = 0.10


@dataclass
class RuntimeConfig:
    """Limits one candidate workflow runs under, per query.

    Attributes:
        max_model_calls: Model calls a single query may make before it is stopped.
        max_tokens: Tokens a single query may spend before it is stopped.
        concurrency: How many examples are scored at once. API latency dominates
            the wall clock, so this is the main speed knob.
        tools: Server-side tools a workflow may call — a subset of
            "code_execution", "web_search", and "web_fetch". A workflow that calls
            one not on this list is rejected, so a closed-book benchmark can set
            `tools: []` and be sure no candidate reached the web. Empty means none.
    """
    max_model_calls: int = 24
    max_tokens: int = 120_000
    concurrency: int = 8
    tools: list[str] = field(default_factory=lambda: ["code_execution", "web_search", "web_fetch"])


@dataclass
class JudgeConfig:
    """Settings for grading free-form answers with a model.

    Attributes:
        model: The model that scores candidate answers against the rubric.
        min_gold_score: An ideal reference answer must score at least this under
            a task-specific rubric, or the rubric is rejected.
        max_empty_score: An empty answer must score at most this, or the rubric
            is rejected. Together these two catch a rubric that doesn't
            discriminate, whose scores would be noise.
    """
    model: str = "claude-haiku-4-5"
    min_gold_score: float = 0.7
    max_empty_score: float = 0.5


@dataclass
class DataConfig:
    """Settings for the dataset: how many examples, and how they split.

    The split is three-way. TRAIN is the only slice the design agent may see —
    its self-test examples and any few-shot material it bakes into prompts —
    carved out first so nothing the agent tunes against is ever scored. DEV
    guides the search (candidates are scored on it, and its failures are fed
    back to later rounds). TEST is held out for the final ranking alone.

    Attributes:
        n_examples: Pool size — how many labeled examples to load or generate —
            when `n_dev`/`n_test` are not set explicitly.
        dev_fraction: Share of the pool (after train is carved out) used as the
            dev split when `n_dev`/`n_test` are not set. The rest is test.
        n_train: Examples the design agent may see and self-test against.
            Disjoint from dev and test.
        n_dev: Explicit dev size. Set BOTH `n_dev` and `n_test` to size the
            splits directly — the pool becomes `n_train + n_dev + n_test` and
            `n_examples`/`dev_fraction` are ignored. 0 uses the fraction.
        n_test: Explicit test size, as above. 0 uses the remainder.
        n_case_types: How many kinds of case to plan up front, so batches can be
            pointed at different ones instead of all writing the typical example.
        batch_size: Examples requested per generation call.
        free_form_batch_size: Smaller batch used when answers are free-form,
            since long answers hit the output ceiling sooner.
        max_stalls: Give up after this many consecutive batches add nothing new.
    """
    n_examples: int = 100
    dev_fraction: float = 0.6
    n_train: int = 5
    n_dev: int = 0
    n_test: int = 0
    n_case_types: int = 15
    batch_size: int = 20
    free_form_batch_size: int = 5
    max_stalls: int = 3


@dataclass
class DesignerConfig:
    """Settings for the agent that writes candidate workflows.

    Attributes:
        model: The model the design agent runs on.
        rounds: How many design rounds to run. Each one costs real money.
        research: Whether to run the web-research phase before designing — a
            Claude Agent SDK session that studies what works for this task online,
            reads as many sources as it needs, and writes `research_notes.md`,
            which is then handed to every design round. Set false to skip it (an
            extra agent session; on by default so designs build on prior art, not
            only what the model already carries in its weights).
        dev_sample_size: Legacy — the train split now serves this purpose (see
            `DataConfig.n_train`). Kept so run configs saved before the split
            existed still load; used only as the fallback sample size when a
            continued run's saved benchmark predates the train split.
        working_skills: Whether the design agent gets a `working_skills/` directory
            it can read and write across the rounds of a single run. A skill
            (a short SKILL.md note on a technique or task gotcha) it writes in one
            round is staged back in for later rounds of the SAME run, then
            discarded when the run ends — a within-run, self-built skill memory.
            Off keeps the fixed skill set.
        failures_shown: How many of each candidate's worst dev examples to hand
            the next round, so it can see WHERE existing designs break — the
            signal a scalar accuracy hides — not just that they scored what they
            scored.
        dominated_shown: How many off-frontier (dominated) candidates to include
            alongside the archive handed to the next round. Every frontier
            candidate is always shown — those are the points a new design must
            beat — but the dominated ones are capped, most-recent first, so the
            prompt stays bounded as the archive grows over rounds. What was
            dropped is stated, never silently truncated.
        skills: Skill directories under `skills/` staged into the agent's
            `.claude/skills/` each round.
        allowed_tools: Tools the agent is permitted to use.
    """
    model: str = "claude-opus-4-8"
    rounds: int = 3
    research: bool = True
    working_skills: bool = False
    dev_sample_size: int = 5
    failures_shown: int = 4
    dominated_shown: int = 10
    skills: list[str] = field(
        default_factory=lambda: ["workflow-design", "workflow-eval", "workflow-naming"])
    allowed_tools: list[str] = field(
        default_factory=lambda: ["WebSearch", "WebFetch", "Bash", "Read", "Write", "Skill"])


@dataclass
class TaskConfig:
    """What you are optimizing for: a description, and optionally your own data.

    `seed_prompt` alone is enough — the analyzer infers everything else. The
    fields after `grader` are escape hatches for a task whose shape you already
    know; setting `description` skips the analyzer's API call entirely.

    Attributes:
        name: Short identifier, used to name the saved run.
        seed_prompt: The task in plain English. The only required field.
        dataset: Path to a `.jsonl` of `{"question", "answer"}` objects, relative
            to the repo root. None means generate one.
        grader: Path to a `.py` exposing `grade(prediction, item) -> float` in
            [0, 1], to score with an external benchmark's own metric.
        description: A ready-made task description. Set it to skip the analyzer.
        check_type: How to grade — "numeric", "exact", or "llm_judge". Only read
            when `description` is set.
        answer_examples: Correctly formatted answers, shown to the design agent
            as the target format. Only read when `description` is set.
        judge_rubric: Grading criteria for `check_type: llm_judge`, so a known-shape
            task can supply its own rubric instead of the analyzer inferring one or
            falling back to a generic judge. Only read when `description` is set and
            `check_type` is "llm_judge"; still calibrated before use.
    """
    name: str = "custom"
    seed_prompt: str = ""
    dataset: Optional[str] = None
    grader: Optional[str] = None
    description: Optional[str] = None
    check_type: Optional[str] = None
    answer_examples: list[str] = field(default_factory=list)
    judge_rubric: str = ""


@dataclass
class ReportConfig:
    """What to report at the end, and where to put it.

    Attributes:
        max_cost_per_query: The budget for the "best workflow I can afford" pick.
        min_accuracy: The floor for the "cheapest workflow that is good enough"
            pick.
        output_dir: Directory the run JSON is written to.
    """
    max_cost_per_query: float = 0.002
    min_accuracy: float = 0.80
    output_dir: str = "runs"


@dataclass
class Config:
    """The whole configuration for one run.

    Attributes:
        task: What is being optimized.
        models: The models the search may use, cheapest first. This one list is
            the search pool, the menu a workflow routes over, and the price table
            cost is measured with.
        analysis_model: The model that analyzes the task and generates data.
        call: Per-call settings.
        runtime: Per-query limits for candidate workflows.
        judge: Free-form grading settings.
        data: Dataset generation settings.
        designer: Design agent settings.
        report: Final reporting settings.
    """
    task: TaskConfig = field(default_factory=TaskConfig)
    models: list[ModelSpec] = field(default_factory=list)
    analysis_model: str = "claude-opus-4-8"
    call: CallConfig = field(default_factory=CallConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    designer: DesignerConfig = field(default_factory=DesignerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


def load_config(task: str = "clinical_notes", overrides: list[str] = ()) -> Any:
    """Build one run's config: base file, then task file, then overrides.

    A task file may set any key, not just `task.*` — a benchmark that needs a
    tighter per-query budget says so next to its own description.

    Args:
        task: Name of a file under `config/task/`, without the extension. Pass
            "" or None to load only the base config.
        overrides: OmegaConf dotlist entries, e.g. `["designer.rounds=1"]`.

    Returns:
        A merged OmegaConf DictConfig validated against `Config`.

    Raises:
        Exception: OmegaConf rejects an unknown key or a value of the wrong type.
    """
    cfg = OmegaConf.merge(
        OmegaConf.structured(Config),
        OmegaConf.load(CONFIG_DIR / "config.yaml"),
    )
    if task:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(CONFIG_DIR / "task" / f"{task}.yaml"))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def load_resolved(path) -> Any:
    """Reload a config that an earlier step already resolved and wrote out whole.

    This is how the design agent's dev evaluator meters candidates exactly as the
    search does, instead of re-deriving the settings and hoping they match.

    Args:
        path: Path to a YAML file written by `OmegaConf.to_yaml(cfg)`.

    Returns:
        A DictConfig validated against `Config`.
    """
    return OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(path))
