"""Getting labeled examples: load the task's own, or generate some.

The generator is more involved than "ask for N examples" for a measured reason.
Each generation call is independent — the model has no memory of earlier batches —
so a naive loop returns the same handful of obvious cases reworded, and near
duplicates don't fail loudly, they just quietly make every accuracy number mean
less. Diversity is therefore engineered three ways: plan the kinds of case up
front and point each batch at different ones, show each batch recent inputs to
avoid, and dedup on a normalized key so paraphrases don't slip through.

NOTE: a Pydantic docstring becomes the JSON Schema "description" and is sent to
the model with the request. Those docstrings are prompt text as well as comments
— keep developer asides in `#` comments, which are not transmitted.
"""
import json
import random
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict

from . import paths, prompts


class LabeledExample(BaseModel):
    """One graded item: the input a workflow sees, and the answer it should return.

    Attributes:
        question: The input given to `solve`.
        answer: The target answer it is graded against.
    """
    model_config = ConfigDict(extra="forbid")
    question: str
    answer: str


class ExampleBatch(BaseModel):
    """What ONE generation call returns — a batch, not the finished dataset.

    Attributes:
        examples: The labeled examples in this batch.
    """
    model_config = ConfigDict(extra="forbid")
    examples: list[LabeledExample]


class CaseTypes(BaseModel):
    """The kinds of case the dataset should cover, planned up front so batches
    don't all reach for the same obvious example. One short phrase per entry, e.g.
    ["division with a remainder", "percentage discount", "rate and distance"] for
    math, or ["discharge summary", "post-op complication"] for clinical notes.

    Attributes:
        case_types: The short phrases naming each kind of case.
    """
    model_config = ConfigDict(extra="forbid")
    case_types: list[str]


def load_examples(path) -> Optional[list[dict]]:
    """Load the task's own labeled examples from a JSONL file.

    Args:
        path: Path to a `.jsonl` of `{"question", "answer"}` objects, relative to
            the repo root or absolute. Falsy means the task brought no data.

    Returns:
        The examples as dicts (extra keys preserved, which a custom grader may
        need), or None when `path` is falsy.

    Raises:
        FileNotFoundError: The path doesn't exist.
    """
    if not path:
        return None
    with open(paths.resolve(path)) as f:
        return [json.loads(line) for line in f if line.strip()]


def take(data: list[dict], n_examples: int, log=print, seed: int = 0) -> list[dict]:
    """Cut a loaded dataset down to the number of examples asked for.

    `n_examples` used to apply only to GENERATED data, so a benchmark that ships
    200 rows ran all 200 however few were asked for — and the cost estimate,
    which sizes itself from `n_examples`, understated the run by the same factor.
    Applying it to both makes the setting mean one thing.

    Sampling is deterministic so two runs at the same size score the same
    examples and their numbers can be compared.

    Args:
        data: The loaded examples.
        n_examples: How many to keep. 0 or less keeps all of them.
        log: Where to note what was taken.
        seed: Seed for the sample.

    Returns:
        Up to `n_examples` examples, or all of them if there are fewer.
    """
    if n_examples <= 0 or len(data) <= n_examples:
        if n_examples > len(data):
            log(f"dataset has {len(data)} examples, fewer than the {n_examples} asked for")
        return data
    log(f"using {n_examples} of {len(data)} examples (random.Random({seed}).sample)")
    return random.Random(seed).sample(data, n_examples)


def generate_examples(cfg, client, analysis, log=print, n_examples=None) -> list[dict]:
    """Generate a diverse labeled dataset for a task that brought no data.

    Generates in SMALL BATCHES — one giant structured-output call runs past
    `max_output_tokens` and comes back as truncated, invalid JSON. See the module
    docstring for why diversity is engineered rather than assumed.

    Args:
        cfg: The run config; reads `cfg.data` and `cfg.analysis_model`.
        client: ModelClient to generate through.
        analysis: The task analysis — its description and check_type decide what
            the answers should look like.
        log: Where progress lines go.
        n_examples: How many examples to target. None reads
            `cfg.data.n_examples`; a caller sizing for explicit splits passes
            the total it actually needs.

    Returns:
        Up to the target of deduplicated `{"question", "answer"}` dicts. May be
        short of the target if generation stalls; the shortfall is logged.
    """
    target = int(cfg.data.n_examples) if n_examples is None else int(n_examples)
    free_form = analysis.check_type == "llm_judge"
    answer_rule = (
        "Each 'answer' is an ideal reference output for that input — it may be "
        "multi-sentence / free-form; it will be graded by an LLM judge."
        if free_form else
        "Each 'answer' must be the correct final target ONLY — a bare value (the "
        "number or the label), with no explanation or units.")
    batch_size = cfg.data.free_form_batch_size if free_form else cfg.data.batch_size

    case_types = _plan_case_types(cfg, client, analysis)
    log(f"generating ~{target} examples across {len(case_types)} "
        f"case types (batches of {batch_size})...")

    data, seen, stalls, case_index = [], set(), 0, 0
    while len(data) < target and stalls < cfg.data.max_stalls:
        chosen = ([case_types[(case_index + j) % len(case_types)] for j in range(3)]
                  if case_types else [])
        case_index += 3
        recent = [item["question"][:80].replace("\n", " ") for item in data[-8:]]

        prompt = prompts.render(
            "generate_examples",
            k=min(batch_size, target - len(data)),
            description=analysis.description,
            answer_rule=answer_rule,
            case_hint=(" Cover these kinds of case specifically: " + "; ".join(chosen) + "."
                       if chosen else ""),
            avoid_hint=("\n\nMake them DIFFERENT from these already-generated inputs:\n- "
                        + "\n- ".join(recent) if recent else ""))
        try:
            batch = client.parse(cfg.analysis_model, prompt, ExampleBatch).examples
        except Exception:
            batch = []                    # truncated / garbled batch -> skip, don't crash

        before = len(data)
        for item in batch:
            key = _normalize(item.question)
            if key and key not in seen:
                seen.add(key)
                # plain dicts, so generated and user-supplied data look the same
                data.append({"question": item.question, "answer": item.answer})
        stalls = stalls + 1 if len(data) == before else 0

    if len(data) < target:
        log(f"(note: generated {len(data)}/{target} unique examples)")
    return data[:target]


def _plan_case_types(cfg, client, analysis) -> list[str]:
    """Ask up front for the kinds of case a good test set spans.

    Batches are then pointed at different ones, instead of each independently
    generating the "typical" example.

    Args:
        cfg: The run config; reads `cfg.data.n_case_types`.
        client: ModelClient to call.
        analysis: The task analysis, for its description.

    Returns:
        Short phrases naming each kind of case, or [] if the call failed — in
        which case generation proceeds without case hints.
    """
    prompt = prompts.render("case_types", k=cfg.data.n_case_types,
                            description=analysis.description)
    try:
        return client.parse(cfg.analysis_model, prompt, CaseTypes).case_types
    except Exception:
        return []


def _normalize(text: str) -> str:
    """Build a key for near-duplicate detection.

    Args:
        text: A generated question.

    Returns:
        The text lowercased with every run of non-alphanumerics collapsed to one
        space, so paraphrases an exact-match check would miss collide here.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
