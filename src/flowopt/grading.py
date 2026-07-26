"""Scoring a workflow's answer against the gold answer.

A workflow RETURNS its answer, so nothing is parsed out of prose on either side:
`numeric` requires the answer to BE a number, and the judge replies under a
schema. Every rule returns a score in [0, 1]; a program's accuracy is the mean
score over the dataset.
"""
import importlib.util
import re
from dataclasses import dataclass
from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict

from . import paths, prompts
from .client import ModelClient


class JudgeScore(BaseModel):
    """The judge's reply: a number, not a sentence containing one.

    Attributes:
        score: Quality of the candidate answer, 0-100.
    """
    model_config = ConfigDict(extra="forbid")
    score: int


def extract_last_number(text) -> Optional[float]:
    """Pull the last number out of free text.

    NOT used by grading. It is handed to workflow programs, which may want to
    read a value out of a free-text intermediate result mid-pipeline.

    Args:
        text: Any text, or None.

    Returns:
        The last number in the text as a float, or None if there isn't one.
    """
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text or "")
    if not numbers:
        return None
    try:
        return float(numbers[-1].replace(",", ""))
    except ValueError:
        return None


def as_number(value) -> Optional[float]:
    """Read a value that must BE a number, not merely contain one.

    "42" and " 1,024 " parse; "42 apples" does not. Searching prose for the last
    number instead would grade "42 out of 100" as 100.

    Args:
        value: The candidate or gold answer.

    Returns:
        The value as a float, or None if it isn't purely a number.
    """
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


@dataclass
class Grader:
    """Scores one predicted answer against the gold answer, in [0, 1].

    `kind` selects the rule:
        "numeric"   — the answer must be the same number. 1.0 or 0.0.
        "exact"     — case-insensitive string match. 1.0 or 0.0.
        "llm_judge" — a graded quality score from a cheap model against a
                      task-specific rubric, so "accuracy" means mean quality.
        "custom"    — an external benchmark's own metric (see `from_grader`),
                      which sees the whole item, not just the gold string.

    The judge's own API calls are the evaluator's cost, deliberately NOT counted
    as workflow cost.

    Attributes:
        kind: Which rule to apply, as above.
        client: The ModelClient the judge calls. Required only for "llm_judge".
        judge_model: Model id used to judge.
        task: Task description, given to the judge for context.
        rubric: Task-specific grading criteria. Empty falls back to a generic
            judge prompt.
        grade_fn: For "custom": `grade(prediction, item) -> float` in [0, 1].
    """
    kind: str
    client: Optional[ModelClient] = None
    judge_model: str = "claude-haiku-4-5"
    task: str = ""
    rubric: str = ""
    grade_fn: Optional[Callable[[str, dict], float]] = None

    @classmethod
    def from_grader(cls, path: str) -> "Grader":
        """Load a task that ships its own metric.

        Args:
            path: Path to a `.py` exposing `grade(prediction, item) -> float` in
                [0, 1], relative to the repo root or absolute.

        Returns:
            A Grader of kind "custom" backed by that function.

        Raises:
            AttributeError: The module defines no `grade`.
            FileNotFoundError: No such file.
        """
        spec = importlib.util.spec_from_file_location("task_grader", paths.resolve(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return cls(kind="custom", grade_fn=module.grade)

    def score(self, prediction, item: dict) -> float:
        """Score one prediction against one dataset item.

        Args:
            prediction: What the workflow returned, already unwrapped to a string.
            item: The dataset example. Needs an "answer" key for every kind but
                "custom", which receives the whole item.

        Returns:
            A score in [0, 1].
        """
        if self.kind == "custom":
            return self.grade_fn(prediction, item)
        gold = item["answer"]
        if self.kind == "numeric":
            predicted, expected = as_number(prediction), as_number(gold)
            return 1.0 if (predicted is not None and expected is not None
                           and abs(predicted - expected) < 1e-6) else 0.0
        if self.kind == "exact":
            return 1.0 if str(prediction).strip().casefold() == str(gold).strip().casefold() else 0.0
        return self.judge(prediction, gold, item.get("question", ""))

    def judge(self, prediction, gold, question="") -> float:
        """Score a free-form answer against the rubric, using a model.

        The gold answer is shown as an example of a good answer, not the only
        acceptable one. The question is shown too, so the judge can tell an answer
        to a DIFFERENT question from a correct one — the gold alone can be
        ambiguous about what was asked.

        Args:
            prediction: The candidate answer.
            gold: A reference answer for the same input.
            question: The input the answer is responding to. "" when there is
                none — e.g. rubric calibration, which judges an answer against
                itself and needs no question.

        Returns:
            A score in [0, 1] — the judge's 0-100 verdict, clamped. Returns 0.0
            if the judge refuses or its reply doesn't parse.
        """
        prompt = prompts.render(
            "judge", task=self.task, question=question,
            rubric=self.rubric.strip() or
                   "Does the candidate correctly and completely satisfy the task?",
            gold=gold, prediction=prediction)
        try:
            score = self.client.parse(self.judge_model, prompt, JudgeScore).score
        except ValueError:                          # refusal, or a reply past the ceiling
            return 0.0
        return max(0.0, min(1.0, score / 100.0))    # a schema can't bound a range, so clamp
