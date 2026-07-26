"""Running a candidate workflow program, metering and capping every model call.

A workflow is a Python program defining `solve(question, call_model) -> answer`.
Because it is ordinary code it can express any inference-time paradigm; the
harness fixes only three things — the contract, the metered call site, and the
grader — and all the generality rides on those.
"""
import builtins
import json
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict

from .grading import Grader, extract_last_number
from .models import ModelCatalog


class Reply(str):
    """What `call_model` hands back to a workflow: the text, with everything attached.

    It IS a string, so `return call_model(prompt)` works and nothing has to be
    unpacked — but nothing is thrown away either.

    Attributes:
        blocks: Every content block the call produced (tool uses, tool results,
            text), in order.
        data: The parsed JSON object when the call passed a `schema`, else None.
            Also None if the model refused or the reply was truncated.
        usage: Token counts for the call.
        model: The model id that actually served the call, after routing.
        truncated: True when the call hit the tool-turn cap mid-work, so the text
            is a partial turn rather than a finished answer.
    """
    def __new__(cls, text, blocks=(), usage=None, model="", data=None, truncated=False):
        """Build a Reply.

        Args:
            text: The reply text; this is the string value of the object.
            blocks: Content blocks from the call.
            usage: Token counts.
            model: Model id that served the call.
            data: Parsed JSON, when a schema was used.
            truncated: Whether the tool-turn cap cut the call off.
        """
        reply = super().__new__(cls, text)
        reply.blocks = list(blocks)
        reply.usage = dict(usage or {})
        reply.model = model
        reply.data = data
        reply.truncated = truncated
        return reply


@dataclass
class CallRecord:
    """One metered model call, kept whole so a run can be inspected afterwards.

    Attributes:
        model: Model id that served the call.
        prompt: The prompt as sent.
        reply: The Reply returned to the workflow.
        cost: USD billed for this call.
    """
    model: str
    prompt: str
    reply: Reply
    cost: float


class Answer(BaseModel):
    """The shape of a final answer.

    Handed to every program as `ANSWER_SCHEMA` so the graded contract and the
    schema a workflow asks the model for cannot drift apart.

    Attributes:
        answer: The final answer, as a string.
    """
    model_config = ConfigDict(extra="forbid")
    answer: str


# Programs receive it as a plain dict: model-written code runs sandboxed without
# pydantic, and `call_model`'s schema= accepts either form.
ANSWER_SCHEMA = Answer.model_json_schema()


class CallMeter:
    """The per-query object whose `call_model` a workflow calls for every model call.

    This is the one place cost is measured and the one place a runaway program is
    stopped. Every call is also recorded, so a run can be inspected afterwards.
    One instance serves one query, so examples share nothing and can run
    concurrently.

    Attributes:
        client: The ModelClient calls are made through.
        default_model: Model used when a workflow names none.
        max_model_calls: Cap on calls for this query.
        max_tokens: Cap on tokens for this query.
        calls: Calls made so far.
        tokens: Tokens spent so far.
        cost: USD spent so far.
        records: A CallRecord per call, in order.
    """

    def __init__(self, client, default_model: str, max_model_calls: int, max_tokens: int,
                 allowed_tools=None):
        """Build a meter for one query.

        Args:
            client: The ModelClient to call through.
            default_model: Model used when the workflow names none.
            max_model_calls: Cap on model calls for this query.
            max_tokens: Cap on tokens for this query.
            allowed_tools: Server-side tools a workflow may call. None means no
                restriction; a list (possibly empty) forbids anything not on it.
        """
        self.client = client
        self.default_model = default_model
        self.max_model_calls = max_model_calls
        self.max_tokens = max_tokens
        self.allowed_tools = allowed_tools
        self.calls = 0
        self.tokens = 0
        self.cost = 0.0
        self.records: list[CallRecord] = []

    def call_model(self, prompt, model=None, system=None, tools=None,
                   effort=None, schema=None) -> Reply:
        """Make one metered model call. This is the function workflows receive.

        Args:
            prompt: The prompt to send.
            model: Model id to route to. Unknown or omitted falls back to the
                default — model-written code routes by name and may invent one.
            system: Optional system prompt.
            tools: Server-side tools to enable: "code_execution", "web_search",
                "web_fetch". A tool not on the task's allowlist raises RuntimeError.
            effort: Thinking depth, "low" through "max". Ignored on models that
                cannot think.
            schema: JSON Schema (or Pydantic class) constraining the reply. The
                parsed object comes back on `reply.data`.

        Returns:
            A Reply: the text, with blocks, parsed data, usage and model attached.

        Raises:
            RuntimeError: This query has exhausted its call or token budget. The
                evaluator catches it and scores the example 0.
        """
        if self.calls >= self.max_model_calls:
            raise RuntimeError("workflow exceeded its model-call budget")
        # Enforced here because this is the one place a workflow reaches a tool.
        # Rejected, not silently dropped: a closed-book run must fail a candidate
        # that tried to search, not quietly answer it a different way.
        if self.allowed_tools is not None:
            for tool in tools or []:
                if tool not in self.allowed_tools:
                    raise RuntimeError(f"tool '{tool}' is not allowed for this task")
        # The web tools bundle their own code-execution sandbox for dynamic
        # filtering; a second one alongside confuses the model, so the API forbids
        # the pair. Reject it here rather than let it 400 mid-search.
        requested = set(tools or [])
        if "code_execution" in requested and requested & {"web_search", "web_fetch"}:
            raise RuntimeError("code_execution cannot be combined with web_search/web_fetch "
                               "in one call — the web tools already run code")
        self.calls += 1
        model = self.client.catalog.resolve(model) if model else self.default_model

        response = self.client.call(model, str(prompt), system=system, tools=tools,
                                    effort=effort, schema=schema)
        data = None
        if schema:
            try:
                data = json.loads(response.text)   # the reply is constrained to the schema
            except ValueError:
                data = None                        # a refusal, or a reply past the ceiling

        cost = self.client.catalog.cost_usd(model, response.usage)
        self.tokens += sum(response.usage.values())
        self.cost += cost

        reply = Reply(response.text, blocks=response.blocks, usage=response.usage,
                      model=model, data=data,
                      truncated=getattr(response, "truncated", False))
        self.records.append(CallRecord(model=model, prompt=str(prompt), reply=reply, cost=cost))

        if self.tokens > self.max_tokens:
            raise RuntimeError("workflow exceeded its token budget")
        return reply


# ---- Compiling a candidate --------------------------------------------------
# Candidate code is model-written, so it always runs with a restricted import list
# and a builtins allowlist rather than full Python. The allowlist has to be
# GENEROUS: a name that's missing doesn't read as "blocked", it reads as "this
# strategy scores 0", and the search then quietly avoids whole families of
# workflow for a reason nothing reports. Anything a plausible workflow writes — a
# helper class, `getattr` on a reply, sampling with `random`, catching
# `RuntimeError` from its own budget — has to work. Only reaching OUT (os, sys,
# subprocess, importlib, open, eval, exec) is off the table.
_ALLOWED_IMPORTS = {"re", "json", "math", "statistics", "collections", "itertools",
                    "functools", "string", "random", "time", "typing", "dataclasses",
                    "textwrap", "operator", "heapq", "difflib", "decimal", "fractions"}
_ALLOWED_BUILTINS = (
    "abs all any bool callable dict divmod enumerate filter float format frozenset "
    "getattr hasattr int isinstance issubclass iter len list map max min next object "
    "pow print range repr reversed round set slice sorted str sum tuple type zip "
    "ArithmeticError AssertionError AttributeError Exception IndexError KeyError "
    "LookupError NameError NotImplementedError OverflowError RuntimeError "
    "StopIteration TypeError ValueError ZeroDivisionError "
    "__build_class__")          # a `class` statement in a candidate needs this


def _guarded_import(name, *args, **kwargs):
    """Stand in for `__import__` inside a candidate, allowing only safe modules.

    Args:
        name: Module being imported.
        *args: Remaining `__import__` arguments.
        **kwargs: Remaining `__import__` keyword arguments.

    Returns:
        The imported module.

    Raises:
        ImportError: The module is not in `_ALLOWED_IMPORTS`.
    """
    if name.split(".")[0] not in _ALLOWED_IMPORTS:
        raise ImportError(f"blocked import in candidate: {name}")
    return builtins.__import__(name, *args, **kwargs)


def compile_solve(code: str, catalog: ModelCatalog, helpers: str = "") -> Callable:
    """Execute a candidate's source and return its `solve` function.

    The program may use `re`, `json`, `statistics`, `Counter`,
    `extract_last_number`, `MODELS` and `ANSWER_SCHEMA` without importing them
    (see the workflow-design skill).

    `helpers`, when given, is executed into the same namespace BEFORE the
    candidate, so anything it defines — the run's shared operators, written by the
    design agent into `working_skills/helpers.py` — is callable from `solve` by
    name, exactly like `extract_last_number`. It runs under the same sandbox as
    the candidate, so a syntax error or a blocked import in the operators fails the
    candidate loudly rather than passing silently.

    This sandbox raises the bar; it is not a security boundary. Run genuinely
    untrusted code in a container.

    Args:
        code: The candidate's Python source.
        catalog: Supplies `MODELS`, the model ids a workflow may route over.
        helpers: Optional operator source to define in the namespace first, so the
            candidate can call its functions. "" injects nothing.

    Returns:
        The program's `solve(question, call_model) -> answer` function.

    Raises:
        ValueError: The source defines no callable `solve`.
        ImportError: The source (or the helpers) imports a module outside the allowlist.
        Exception: Anything else raised while executing the source at module level.
    """
    allowed = {n: getattr(builtins, n) for n in _ALLOWED_BUILTINS.split()}
    allowed["__import__"] = _guarded_import
    namespace = {
        "__builtins__": allowed,
        "__name__": "candidate",     # a `class` statement reads it for __module__
        "re": re, "json": json, "statistics": statistics, "Counter": Counter,
        "extract_last_number": extract_last_number,
        "MODELS": catalog.ids, "ANSWER_SCHEMA": ANSWER_SCHEMA,
    }
    if helpers:
        exec(helpers, namespace)     # the run's shared operators, same sandbox
    exec(code, namespace)
    if not callable(namespace.get("solve")):
        raise ValueError("program does not define solve(question, call_model)")
    return namespace["solve"]


def unwrap_answer(returned) -> str:
    """Reduce whatever `solve()` returned to the string that gets graded.

    A program may return the answer directly, a dict carrying the answer plus
    anything else it wants to keep, or a schema-constrained Reply as-is; all
    three unwrap to the answer.

    Args:
        returned: The value `solve()` handed back.

    Returns:
        The answer as a stripped string.

    Raises:
        ValueError: A dict or parsed reply with no "answer" key. That is a
            contract violation, not an answer — stringifying it would grade
            `{'result': 'positive'}` as the prediction and silently score 0, so
            it raises and the record shows why.
    """
    if isinstance(returned, Reply) and returned.data is not None:
        returned = returned.data
    if isinstance(returned, dict):
        if "answer" not in returned:
            raise ValueError(f"solve() returned an object with no 'answer' key: {sorted(returned)}")
        returned = returned["answer"]
    return str(returned).strip()


# ---- Evaluating a candidate over a dataset ----------------------------------
@dataclass
class SplitScore:
    """How one candidate workflow did on one dataset split.

    Attributes:
        name: The candidate's name.
        accuracy: Mean score over the split, in [0, 1].
        cost: Mean USD per query over the split.
        cached_input_frac: Share of input tokens served from the prompt cache.
        records: One dict per example — question, gold, answer, score, cost,
            error, and `calls` (the full CallRecord trace). Kept for inspection;
            never graded.
        error: Set only when the program didn't compile, in which case the split
            was never run.
    """
    name: str
    accuracy: float = 0.0
    cost: float = 0.0
    cached_input_frac: float = 0.0
    records: list = field(default_factory=list)
    error: str = ""

    @property
    def errors(self) -> list[str]:
        """Per-example error messages, for the examples that raised."""
        return [r["error"] for r in self.records if r["error"]]


class Evaluator:
    """Runs candidate workflows against a dataset and scores them.

    The same metered runtime and the same grader serve dev, test, and the design
    agent's own self-tests, so a number means the same thing wherever it came from.

    Attributes:
        client: The ModelClient workflows call through.
        grader: The Grader scoring returned answers.
        cfg: A `RuntimeConfig` — per-query caps and concurrency.
        default_model: Model a workflow starts on.
    """

    def __init__(self, client, grader: Grader, runtime_cfg, default_model: Optional[str] = None):
        """Build an evaluator.

        Args:
            client: The ModelClient to run workflows through.
            grader: Scores each returned answer.
            runtime_cfg: A `RuntimeConfig`.
            default_model: Model workflows start on. Defaults to the cheapest in
                the client's catalog.
        """
        self.client = client
        self.grader = grader
        self.cfg = runtime_cfg
        self.default_model = default_model or client.catalog.default

    def run(self, program: dict, dataset: list[dict]) -> SplitScore:
        """Run one candidate over a dataset and score it.

        Each example gets its own CallMeter, so examples share nothing and run
        concurrently — the wall clock is otherwise sequential API latency. An
        example whose program crashes or exhausts its budget scores 0 rather than
        sinking the whole evaluation.

        Args:
            program: `{"name": str, "code": str}`, optionally with `"helpers": str`
                — operator source injected before the code (see `compile_solve`).
            dataset: Examples, each `{"question": ..., "answer": ...}`. A custom
                grader may read other keys too.

        Returns:
            A SplitScore. If the program didn't compile, `error` is set and
            accuracy and cost are 0.

        Raises:
            ValueError: `dataset` is empty. Scoring nothing would report accuracy
                0.00 / $0.00000, which is indistinguishable from a measurement.
        """
        if not dataset:
            raise ValueError(f"nothing to evaluate {program['name']} on: empty dataset")
        try:
            solve = compile_solve(program["code"], self.client.catalog,
                                  helpers=program.get("helpers", ""))
        except Exception as error:
            return SplitScore(name=program["name"], error=f"compile: {error}")

        workers = min(self.cfg.concurrency, len(dataset))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(lambda item: self._run_one(solve, item), dataset))
        else:
            records = [self._run_one(solve, item) for item in dataset]

        return SplitScore(
            name=program["name"],
            accuracy=statistics.mean(r["score"] for r in records),
            cost=statistics.mean(r["cost"] for r in records),
            cached_input_frac=_cached_input_frac(records),
            records=records,
        )

    def _run_one(self, solve: Callable, item: dict) -> dict:
        """Run and grade a candidate on a single example.

        Args:
            solve: The compiled `solve` function.
            item: One dataset example.

        Returns:
            A record dict: question, gold, answer, score, cost, error (None when
            it ran cleanly), and `calls` — the CallRecord trace.
        """
        meter = CallMeter(self.client, self.default_model,
                          self.cfg.max_model_calls, self.cfg.max_tokens,
                          allowed_tools=list(getattr(self.cfg, "tools", []) or []))
        answer, score, error = "", 0.0, None
        try:
            answer = unwrap_answer(solve(item["question"], meter.call_model))
            score = self.grader.score(answer, item)
        except Exception as failure:
            error = f"{type(failure).__name__}: {failure}"
        return {"question": item["question"], "gold": item.get("answer", ""),
                "answer": answer, "score": score, "cost": meter.cost,
                "error": error, "calls": meter.records}


def _cached_input_frac(records: list[dict]) -> float:
    """Compute what share of input tokens came from the prompt cache.

    Prompt caching silently does nothing below a per-model size floor (~4k tokens
    on haiku/opus, ~1k on sonnet), so a workflow built on "resending is cheap" can
    be paying full price with no error to say so. Surfaced here rather than left
    to be discovered from the bill.

    Args:
        records: Per-example records from a run.

    Returns:
        Cached share of input tokens in [0, 1]; 0.0 when no input was billed.
    """
    fresh = cached = 0
    for record in records:
        for call in record["calls"]:
            fresh += call.reply.usage["input"] + call.reply.usage["cache_write"]
            cached += call.reply.usage["cache_read"]
    total = fresh + cached
    return cached / total if total else 0.0
