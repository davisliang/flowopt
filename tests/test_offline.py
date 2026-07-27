"""Everything that can be checked without spending money.

The API is faked, so this covers the parts that are pure logic: pricing, grading,
the Pareto helpers, the metered runtime's guardrails, and what the design agent
is handed. Run with `uv run pytest`.
"""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flowopt import analysis, prompts
from flowopt.analysis import Benchmark, TaskAnalysis
from flowopt.config import load_config, load_resolved
from flowopt.designer import _round_prompt, _stage_agent_dir, summarize_archive
from flowopt.grading import Grader, as_number, extract_last_number
from flowopt.client import ApiResponse, ModelClient
from flowopt.models import ModelCatalog
from flowopt.optimizer import DEV, TEST, Candidate
from flowopt.pareto import best_under_budget, cheapest_above_accuracy, pareto_front
from flowopt.paths import ROOT, SKILLS_DIR
from flowopt.runtime import Evaluator, SplitScore, compile_solve, unwrap_answer
from flowopt.session import Session

SRC = ROOT / "src"

DATA = [{"question": "add 2 and 3", "answer": "5"},
        {"question": "add 10 and 5", "answer": "15"}]


@pytest.fixture
def cfg():
    return load_config("gsm8k")


@pytest.fixture
def catalog(cfg):
    return ModelCatalog.from_config(cfg)


class FakeClient:
    """Answers with the sum of the digits in the prompt, and bills a fixed usage."""

    def __init__(self, catalog):
        self.catalog = catalog

    def call(self, model, prompt, system=None, tools=None, effort=None, schema=None):
        answer = str(sum(int(t) for t in prompt.split() if t.isdigit()))
        return ApiResponse(text=json.dumps({"answer": answer}) if schema else answer,
                    usage={"input": 100, "output": 10, "cache_write": 0, "cache_read": 0})


def evaluator(cfg, catalog, grader=None):
    return Evaluator(FakeClient(catalog), grader or Grader(kind="numeric"), cfg.runtime)


# ---- the OpenRouter client --------------------------------------------------
def _or_response(text="ok", cached=0, finish="stop", cost=None):
    """A minimal stand-in for an OpenRouter chat completion."""
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=text, annotations=None),
            finish_reason=finish)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10,
                              prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
                              cost=cost))


class FakeOpenRouter:
    """Serves one canned completion and keeps the request for inspection."""

    def __init__(self, response):
        self._response = response
        self.request = None
        self.chat = self            # the client reaches it as chat.completions.create
        self.completions = self

    def create(self, **request):
        self.request = request
        return self._response


def test_calls_go_through_openrouter_with_cached_tokens_carved_out(cfg, catalog):
    # OpenRouter counts cached tokens inside prompt_tokens; they are split out so
    # the cache-read price applies to exactly them
    sdk = FakeOpenRouter(_or_response(text="42", cached=30, cost=0.00123))
    response = ModelClient(catalog, cfg.call, client=sdk).call(
        "anthropic/claude-haiku-4.5", "q", tools=["web_search", "web_fetch"])
    assert response.text == "42"
    assert response.usage == {"input": 70, "output": 10, "cache_write": 0, "cache_read": 30}
    assert response.cost == 0.00123                       # what OpenRouter billed
    assert sdk.request["model"] == "anthropic/claude-haiku-4.5"
    # OpenRouter server tools, in the tools array
    assert sdk.request["tools"] == [{"type": "openrouter:web_search"},
                                    {"type": "openrouter:web_fetch"}]
    # no thinking unless the workflow asks — provider defaults would bill it
    assert sdk.request["extra_body"]["reasoning"] == {"effort": "none"}


def test_the_legacy_web_name_and_request_plugins(cfg, catalog):
    # "web" (the deprecated one-plugin spelling) folds onto web_search, and
    # call.plugins rides along on every request
    cfg.call.plugins = ["response-healing"]
    sdk = FakeOpenRouter(_or_response())
    ModelClient(catalog, cfg.call, client=sdk).call(
        "anthropic/claude-haiku-4.5", "q", tools=["web", "web_search"])
    assert sdk.request["tools"] == [{"type": "openrouter:web_search"}]   # deduped
    assert sdk.request["extra_body"]["plugins"] == [{"id": "response-healing"}]


def test_effort_passes_through_and_length_is_truncated(cfg, catalog):
    sdk = FakeOpenRouter(_or_response(finish="length"))
    response = ModelClient(catalog, cfg.call, client=sdk).call(
        "openai/gpt-5.6-sol", "q", effort="max")
    assert sdk.request["extra_body"]["reasoning"] == {"effort": "max"}
    assert response.truncated is True


def test_the_prompt_carries_a_cache_breakpoint(cfg, catalog):
    sdk = FakeOpenRouter(_or_response())
    ModelClient(catalog, cfg.call, client=sdk).call(
        "anthropic/claude-sonnet-5", "q", system="s")
    for message in sdk.request["messages"]:
        assert message["content"][0]["cache_control"] == {"type": "ephemeral"}


# ---- catalog and pricing ----------------------------------------------------
def usage(**kw):
    return {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0, **kw}


def test_catalog_is_ordered_cheapest_first(catalog):
    # sorted by fetched OpenRouter prices, not by config order
    assert catalog.ids == ["anthropic/claude-haiku-4.5", "openai/gpt-5.6-luna",
                           "anthropic/claude-sonnet-5", "openai/gpt-5.6-terra",
                           "anthropic/claude-opus-4.8", "openai/gpt-5.6-sol"]
    assert catalog.default == "anthropic/claude-haiku-4.5"


def test_catalog_carries_artificial_analysis_measurements(catalog):
    # AA orders haiku's slug differently ("claude-4-5-haiku") — the token-set
    # fallback still finds it, so every default-pool model is annotated
    for spec in catalog.specs:
        assert spec.aa is not None, spec.id
    described = catalog.describe("openai/gpt-5.6-luna")
    assert "intelligence" in described and "tok/s" in described


def test_unknown_model_falls_back_to_default(catalog):
    # model-written code routes by name and may invent one
    assert catalog.resolve("gpt-9") == catalog.default


def test_cache_changes_the_price(catalog):
    # the same 2,000 haiku input tokens, three ways: fresh, first send, resent —
    # cache prices come per-model from OpenRouter ($1.25/M write, $0.10/M read)
    haiku = "anthropic/claude-haiku-4.5"
    assert catalog.cost_usd(haiku, usage(input=2000)) == pytest.approx(0.0020)
    assert catalog.cost_usd(haiku, usage(cache_write=2000)) == pytest.approx(0.0025)
    assert catalog.cost_usd(haiku, usage(cache_read=2000)) == pytest.approx(0.0002)


# ---- grading ----------------------------------------------------------------
def test_a_numeric_answer_must_be_the_number():
    assert as_number("42") == 42.0
    assert as_number(" 1,024 ") == 1024.0          # thousands separators are fine
    assert as_number("42 apples") is None          # an answer must BE the number
    check = Grader(kind="numeric")
    assert check.score("42", {"answer": 42}) == 1.0
    assert check.score("42 apples", {"answer": "42"}) == 0.0


def test_exact_check_ignores_case():
    check = Grader(kind="exact")
    assert check.score("Positive", {"answer": "positive"}) == 1.0
    assert check.score("cat", {"answer": "dog"}) == 0.0


def test_extract_last_number_is_for_programs_not_grading():
    assert extract_last_number("Sold 75 of 84 apples, so 9 remain.") == 9.0
    assert extract_last_number("none here") is None


# ---- the answer contract ----------------------------------------------------
def test_unwrap_answer_reduces_what_solve_returned():
    assert unwrap_answer("42") == "42"
    assert unwrap_answer({"answer": " 42 ", "note": "kept"}) == "42"


def test_an_object_with_no_answer_key_is_a_contract_violation():
    # stringifying it would silently grade "{'result': 'positive'}" and score 0
    with pytest.raises(ValueError):
        unwrap_answer({"result": "positive"})


# ---- pareto -----------------------------------------------------------------
def results():
    return [SplitScore("cheap", 0.70, 0.0002), SplitScore("cot", 0.90, 0.0010),
            SplitScore("sc", 0.92, 0.0050), SplitScore("dominated", 0.80, 0.0020)]


def test_pareto_front_drops_dominated_results():
    # "dominated" goes: "cot" is both cheaper AND more accurate
    assert [r.name for r in pareto_front(results())] == ["cheap", "cot", "sc"]


def test_the_two_constrained_picks():
    assert best_under_budget(results(), 0.0015).name == "cot"
    assert cheapest_above_accuracy(results(), 0.90).name == "cot"
    assert best_under_budget(results(), 0.0) is None
    assert cheapest_above_accuracy(results(), 1.0) is None


def test_a_search_reports_its_frontier_on_test_scores():
    from flowopt.optimizer import Search

    # B is dominated on dev but not on test — the reported frontier must follow test
    candidates = [
        Candidate("A", "", "a", dev=SplitScore("A", 0.9, 0.001), test=SplitScore("A", 0.5, 0.001)),
        Candidate("B", "", "b", dev=SplitScore("B", 0.5, 0.002), test=SplitScore("B", 0.9, 0.002)),
    ]
    search = Search(archive=candidates, finalists=candidates)
    assert [c.name for c in search.test_frontier()] == ["A", "B"]


def test_the_split_being_compared_on_is_the_callers_choice():
    # same candidates, opposite verdicts — which is exactly why the caller names
    # the split instead of the object picking one
    candidates = [
        Candidate("A", "", "a", dev=SplitScore("A", 0.9, 0.001), test=SplitScore("A", 0.5, 0.001)),
        Candidate("B", "", "b", dev=SplitScore("B", 0.5, 0.002), test=SplitScore("B", 0.9, 0.002)),
    ]
    assert [c.name for c in pareto_front(candidates, on=DEV)] == ["A"]
    assert [c.name for c in pareto_front(candidates, on=TEST)] == ["A", "B"]
    assert best_under_budget(candidates, 0.005, on=DEV).name == "A"
    assert best_under_budget(candidates, 0.005, on=TEST).name == "B"


# ---- the metered runtime ----------------------------------------------------
GOOD = {"name": "H", "code": (
    "def solve(question, call_model):\n"
    "    reply = call_model(question, schema=ANSWER_SCHEMA)\n"
    "    return reply.data['answer'] if reply.data else str(reply).strip()\n")}


def test_a_working_candidate_is_scored_and_metered(cfg, catalog):
    result = evaluator(cfg, catalog).run(GOOD, DATA)
    assert result.accuracy == 1.0
    assert result.cost > 0
    assert not result.errors
    assert len(result.records[0]["calls"]) == 1     # the call trace is kept


def test_the_sandbox_blocks_imports_outside_the_allowlist(cfg, catalog):
    sneaky = {"name": "bad", "code": "import os\ndef solve(q, c):\n    return os.getcwd()\n"}
    assert "compile" in evaluator(cfg, catalog).run(sneaky, DATA).error


def test_a_crash_scores_zero_instead_of_sinking_the_run(cfg, catalog):
    crasher = {"name": "boom", "code": "def solve(q, c):\n    raise ValueError('x')\n"}
    result = evaluator(cfg, catalog).run(crasher, DATA)
    assert result.accuracy == 0.0
    assert len(result.errors) == len(DATA)


def test_a_runaway_program_hits_its_call_budget(cfg, catalog):
    runaway = {"name": "loop", "code": (
        "def solve(q, call_model):\n"
        "    for _ in range(100):\n        call_model(q)\n    return '0'\n")}
    assert "budget" in evaluator(cfg, catalog).run(runaway, DATA).errors[0]


def test_a_program_without_solve_is_rejected(cfg, catalog):
    assert "solve" in evaluator(cfg, catalog).run({"name": "x", "code": "y = 1\n"}, DATA).error


def test_operators_are_injected_so_a_workflow_can_call_them(catalog):
    # a function defined in `helpers` is callable from `solve` by name, no import —
    # this is how a workflow uses the operators the design agent wrote for the run
    helpers = "def triple(x):\n    return x * 3\n"
    code = "def solve(question, call_model):\n    return str(triple(int(question)))\n"
    assert compile_solve(code, catalog, helpers=helpers)("14", None) == "42"


def test_a_broken_operator_fails_the_candidate_not_the_search(cfg, catalog):
    # a syntax error in the shared operators surfaces as a compile error on the
    # candidate, caught like any other, rather than crashing the run
    program = {"name": "x", "code": "def solve(q, c):\n    return '1'\n",
               "helpers": "def bad(:\n    pass\n"}
    assert "compile" in evaluator(cfg, catalog).run(program, DATA).error


def test_a_candidate_carries_its_operators_into_its_program():
    c = Candidate("H", "", "def solve(q, c): return '1'", helpers="def op(): pass")
    assert c.program["helpers"] == "def op(): pass"


# The sandbox has to admit ordinary Python. A name it refuses doesn't read as
# "blocked" in the results — it reads as "this strategy scores 0", and the search
# then avoids a whole family of workflow for a reason nothing reports.
ORDINARY = {
    "sampling": "import random\ndef solve(q, m):\n    return str(random.choice([5, 15]))\n",
    "a helper class": ("class V:\n    def __init__(self, x):\n        self.x = x\n"
                       "def solve(q, m):\n    return V('5').x\n"),
    "getattr on a reply": ("def solve(q, call_model):\n"
                           "    r = call_model(q, schema=ANSWER_SCHEMA)\n"
                           "    return (getattr(r, 'data', None) or {}).get('answer', '5')\n"),
    "catching its own budget error": ("def solve(q, call_model):\n"
                                      "    try:\n        return str(call_model(q))\n"
                                      "    except RuntimeError:\n        return '0'\n"),
    "typing hints": "from typing import Optional\ndef solve(q, m) -> Optional[str]:\n    return '5'\n",
}


@pytest.mark.parametrize("what", list(ORDINARY))
def test_the_sandbox_admits_ordinary_python(cfg, catalog, what):
    result = evaluator(cfg, catalog).run({"name": what, "code": ORDINARY[what]}, DATA)
    assert not result.error, f"{what}: {result.error}"
    assert not result.errors, f"{what}: {result.errors}"


def test_an_empty_dataset_raises_instead_of_scoring_zero(cfg, catalog):
    # accuracy 0.00 / $0.00000 is indistinguishable from a real measurement
    with pytest.raises(ValueError):
        evaluator(cfg, catalog).run(GOOD, [])


@pytest.mark.parametrize("reach_out", [
    "def solve(q, m):\n    return open('/etc/passwd').read()\n",
    "def solve(q, m):\n    return eval('1+1')\n",
    "import subprocess\ndef solve(q, m):\n    return '5'\n",
])
def test_the_sandbox_still_blocks_reaching_outside(cfg, catalog, reach_out):
    result = evaluator(cfg, catalog).run({"name": "x", "code": reach_out}, DATA)
    assert result.error or result.errors     # at compile time or at call time


def test_the_eval_skill_always_emits_one_json_line(tmp_path):
    # the agent parses this; a traceback would leave it guessing
    script = SKILLS_DIR / "workflow-eval" / "eval_candidate.py"
    for args in ([], ["nope.py"]):
        done = subprocess.run([sys.executable, str(script), *args], cwd=tmp_path,
                              capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": str(SRC)})
        assert json.loads(done.stdout.strip())["ok"] is False, done.stdout + done.stderr


# ---- config -----------------------------------------------------------------
def test_overrides_win_over_the_task_file():
    cfg = load_config("gsm8k", ["designer.rounds=1", "runtime.concurrency=2"])
    assert (cfg.designer.rounds, cfg.runtime.concurrency) == (1, 2)
    assert cfg.task.name == "gsm8k"                 # untouched keys survive


def test_an_unknown_key_is_rejected():
    with pytest.raises(Exception):
        load_config("gsm8k", ["designer.rouds=1"])


def test_a_task_can_supply_its_own_judge_rubric():
    # a known-shape task carries its rubric straight through, skipping the analyzer
    judged = analysis.analysis_from_config(load_config("fanoutqa_judge"))
    assert judged.check_type == "llm_judge"
    assert "reference" in judged.judge_rubric.lower()      # the task's own rubric, not generic
    # a described task that sets no rubric falls back to empty (the generic judge)
    assert analysis.analysis_from_config(load_config("game24")).judge_rubric == ""


def test_a_session_wires_config_catalog_and_client_together(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-used")
    session = Session.load("gsm8k", ["designer.rounds=1"])
    assert session.cfg.designer.rounds == 1
    assert session.catalog.ids[0] == "anthropic/claude-haiku-4.5"
    assert session.client.catalog is session.catalog          # one catalog, not two
    assert session.evaluator(Grader(kind="numeric")).default_model == session.catalog.default


# ---- prompts ----------------------------------------------------------------
def test_prompts_are_filled_from_files():
    text = prompts.render("judge", task="t", question="q", rubric="r", gold="g", prediction="p")
    assert "Task: t" in text and "Question:" in text and "$" not in text


def test_a_missing_prompt_variable_is_an_error_not_a_blank():
    with pytest.raises(KeyError):
        prompts.render("judge", task="t")


# ---- the train / dev / test split -------------------------------------------
def many(n, prefix="q"):
    return [{"question": f"{prefix}{i}", "answer": str(i)} for i in range(n)]


def test_the_split_is_three_way_and_disjoint():
    cfg = load_config("gsm8k", ["data.n_examples=40", "data.n_train=5"])
    train, dev, test = analysis.split_examples(cfg, many(100), log=lambda *a: None)
    assert len(train) == 5
    assert len(train) + len(dev) + len(test) == 40
    questions = [r["question"] for r in train + dev + test]
    assert len(questions) == len(set(questions))          # disjoint — nothing shared


def test_explicit_split_sizes_win_over_the_fraction():
    cfg = load_config("gsm8k", ["data.n_train=3", "data.n_dev=10", "data.n_test=7"])
    train, dev, test = analysis.split_examples(cfg, many(100), log=lambda *a: None)
    assert (len(train), len(dev), len(test)) == (3, 10, 7)


def test_the_split_is_deterministic():
    cfg = load_config("gsm8k", ["data.n_examples=30"])
    first = analysis.split_examples(cfg, many(100), log=lambda *a: None)
    second = analysis.split_examples(cfg, many(100), log=lambda *a: None)
    assert [[r["question"] for r in part] for part in first] == \
           [[r["question"] for r in part] for part in second]


def test_allocated_holdout_examples_become_the_test_split():
    """routerllm's baselines were measured on its holdout; rows labeled
    split=test are that allocation, and OUR test split must be exactly them —
    otherwise a comparison against the recorded baselines compares different
    examples."""
    cfg = load_config("gsm8k", ["data.n_examples=20", "data.n_train=2"])
    data = many(50) + [{"question": f"h{i}", "answer": "x", "split": "test"} for i in range(8)]
    train, dev, test = analysis.split_examples(cfg, data, log=lambda *a: None)
    assert sorted(r["question"] for r in test) == [f"h{i}" for i in range(8)]
    assert all(not r["question"].startswith("h") for r in train + dev)
    # n_examples budgets the whole pool: 20 total - 8 allocated test - 2 train
    assert len(train) == 2 and len(dev) == 10


def test_each_partition_draws_only_from_its_allocated_counterpart():
    """With a full routerllm allocation, our train samples their train, dev
    their val, test their test — an example never crosses a partition."""
    cfg = load_config("gsm8k", ["data.n_train=3", "data.n_dev=4", "data.n_test=5"])
    data = ([{"question": f"tr{i}", "answer": "x", "split": "train"} for i in range(50)]
            + [{"question": f"v{i}", "answer": "x", "split": "val"} for i in range(10)]
            + [{"question": f"te{i}", "answer": "x", "split": "test"} for i in range(8)])
    train, dev, test = analysis.split_examples(cfg, data, log=lambda *a: None)
    assert (len(train), len(dev), len(test)) == (3, 4, 5)
    assert all(r["split"] == "train" for r in train)
    assert all(r["split"] == "val" for r in dev)
    assert all(r["split"] == "test" for r in test)
    # blank sizes take each partition whole, still never crossing
    cfg = load_config("gsm8k", ["data.n_train=3"])
    train, dev, test = analysis.split_examples(cfg, data, log=lambda *a: None)
    assert (len(dev), len(test)) == (10, 8)


def test_too_few_examples_raise_instead_of_an_empty_split():
    cfg = load_config("gsm8k")
    with pytest.raises(ValueError):
        analysis.split_examples(cfg, many(1), log=lambda *a: None)


# ---- what the design agent is handed ----------------------------------------
def benchmark_fixture():
    analysis = TaskAnalysis(description="Add two numbers.", check_type="numeric",
                            judge_rubric="", answer_examples=["5", "15"])
    return Benchmark(analysis=analysis, grader=Grader(kind="numeric"), dev=DATA, test=DATA)


def test_the_agent_dir_has_everything_the_eval_skill_reads(cfg, tmp_path):
    benchmark = benchmark_fixture()
    benchmark.train = [{"question": "add 1 and 1", "answer": "2"}]
    _stage_agent_dir(cfg, benchmark, tmp_path)
    assert json.loads((tmp_path / "task_spec.json").read_text())["check"]["type"] == "numeric"
    # the agent self-tests on TRAIN — dev stays unseen so its scores are honest
    assert json.loads((tmp_path / "train_task.json").read_text()) == benchmark.train


def test_an_old_benchmark_without_train_falls_back_to_a_dev_sample(cfg, tmp_path):
    # continuations of runs saved before the train split existed still stage
    # something for the agent to self-test on — the old behavior, leak and all
    _stage_agent_dir(cfg, benchmark_fixture(), tmp_path)   # fixture has no train
    assert json.loads((tmp_path / "train_task.json").read_text()) == DATA
    for skill in cfg.designer.skills:
        assert (tmp_path / ".claude" / "skills" / skill / "SKILL.md").exists()
    # the staged config round-trips, so the agent meters candidates as the search does
    assert load_resolved(tmp_path / "run_config.yaml").runtime.max_model_calls == cfg.runtime.max_model_calls


def test_working_skills_off_by_default_stages_nothing_extra(cfg, tmp_path):
    _stage_agent_dir(cfg, benchmark_fixture(), tmp_path)   # cfg default: working_skills off
    assert not (tmp_path / "working_skills").exists()
    assert not (tmp_path / ".claude" / "skills" / "workflow-skills").exists()


def test_working_skills_are_staged_for_reading_and_collected_after(tmp_path):
    from flowopt.designer import _collect_skills
    cfg = load_config("gsm8k", ["designer.working_skills=true"])
    run_skills = tmp_path / "run_skills"
    (run_skills / "prior").mkdir(parents=True)
    (run_skills / "prior" / "SKILL.md").write_text("---\nname: prior\n---\na lesson")

    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    _stage_agent_dir(cfg, benchmark_fixture(), agent_dir, run_skills)
    # an earlier round's skill is handed to the agent, and the meta-skill is loaded
    assert (agent_dir / "working_skills" / "prior" / "SKILL.md").exists()
    assert (agent_dir / ".claude" / "skills" / "workflow-skills" / "SKILL.md").exists()

    # a skill the agent writes this round is persisted back for the next round,
    # and the round reports the full inventory rather than leaving it to be
    # discovered from the filesystem
    learned = agent_dir / "working_skills" / "learned"
    learned.mkdir()
    (learned / "SKILL.md").write_text("---\nname: learned\n---\nstrip trailing whitespace")
    (agent_dir / "working_skills" / "helpers.py").write_text("def op():\n    pass\n")
    built = _collect_skills(agent_dir, run_skills)
    assert (run_skills / "learned" / "SKILL.md").exists()
    assert built == ["learned", "prior", "helpers.py"]     # notes sorted, operators last


def test_the_meta_skill_in_the_skill_list_is_not_staged_twice(tmp_path):
    # the UI lets a user pick skills AND flip working_skills on; if they pick
    # workflow-skills explicitly, the second copytree used to FileExistsError
    cfg = load_config("gsm8k", ["designer.working_skills=true",
                                "designer.skills=[workflow-design,workflow-skills]"])
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    _stage_agent_dir(cfg, benchmark_fixture(), agent_dir, tmp_path / "rs")   # must not raise
    assert (agent_dir / ".claude" / "skills" / "workflow-skills" / "SKILL.md").exists()


def test_round_one_asks_for_diversity_and_later_rounds_extend_the_frontier(cfg):
    benchmark = benchmark_fixture()
    first = _round_prompt(cfg, benchmark, 1, "")
    assert "DIVERSE" in first and "anthropic/claude-haiku-4.5" in first
    assert "Artificial Analysis" in first          # the menu carries measurements
    assert "must BE the number" in first          # the strict checker is spelled out

    archive = [Candidate("H", "", "code-a", dev=SplitScore("H", 0.8, 0.001))]
    later = _round_prompt(cfg, benchmark, 2, summarize_archive(archive))
    assert "Pareto frontier" in later and "code-a" in later     # extend, not just cheapen
    assert "ON FRONTIER" in later                               # its one candidate is on it


def test_tool_log_lines_say_what_the_tool_did():
    # "[tool] Bash" tells a reader nothing; the command is the content
    import os as _os
    from flowopt.proposer import _tool_line

    bash = SimpleNamespace(name="Bash",
                           input={"command": "python eval_candidate.py c1.py\n",
                                  "description": "eval"})
    assert _tool_line(bash) == "  [tool] Bash: python eval_candidate.py c1.py"

    # file tools carry the path and the size of what happened; the agent's own
    # scratch prefix (this process's cwd) is stripped as noise
    write = SimpleNamespace(name="Write", input={
        "file_path": _os.path.join(_os.getcwd(), "cand_H.py"), "content": "x" * 1234})
    assert _tool_line(write) == "  [tool] Write: cand_H.py (1234 chars)"
    edit = SimpleNamespace(name="Edit", input={
        "file_path": "cand_H.py", "old_string": "ab", "new_string": "abcd"})
    assert _tool_line(edit) == "  [tool] Edit: cand_H.py (-2 +4 chars)"
    read = SimpleNamespace(name="Read", input={"file_path": "task_spec.json"})
    assert _tool_line(read) == "  [tool] Read: task_spec.json"

    # a tool whose input has no string field still shows its arguments, as JSON
    odd = SimpleNamespace(name="TodoWrite", input={"todos": [{"t": "step 1"}]})
    assert '"step 1"' in _tool_line(odd)
    assert _tool_line(SimpleNamespace(name="X", input={})) == "  [tool] X"

    long = SimpleNamespace(name="Bash", input={"command": "x" * 500})
    assert len(_tool_line(long)) < 330 and _tool_line(long).endswith("…")


# ---- continuing a finished search -------------------------------------------
def test_a_benchmark_round_trips_through_its_saved_form(cfg, catalog):
    # a continued run must score the SAME splits — generated data can't be
    # regenerated identically, so the benchmark is saved whole and rebuilt
    saved = json.loads(json.dumps(analysis.benchmark_to_dict(benchmark_fixture())))
    rebuilt = analysis.benchmark_from_dict(cfg, FakeClient(catalog), saved)
    assert rebuilt.dev == DATA and rebuilt.test == DATA
    assert rebuilt.grader.kind == "numeric"
    assert rebuilt.description == "Add two numbers."


def test_a_saved_judge_rubric_survives_without_recalibration(cfg, catalog):
    judged = Benchmark(
        analysis=TaskAnalysis(description="Summarize.", check_type="llm_judge",
                              judge_rubric="unused-precalibration", answer_examples=[]),
        grader=Grader(kind="llm_judge", client=None, task="Summarize.",
                      rubric="the POST-calibration rubric"),
        dev=DATA, test=DATA, judge_status="ok")
    rebuilt = analysis.benchmark_from_dict(
        cfg, FakeClient(catalog), analysis.benchmark_to_dict(judged))
    assert rebuilt.grader.kind == "llm_judge"
    assert rebuilt.grader.rubric == "the POST-calibration rubric"   # not re-inferred
    assert rebuilt.judge_status == "ok"


def test_rebuild_search_restores_the_archive_and_its_failures():
    from flowopt.optimizer import rebuild_search

    result = {"candidates": [
        {"name": "H", "description": "one call", "code": "code-h",
         "dev": {"accuracy": 0.5, "cost_per_query": 0.001, "cached_input_frac": 0.2},
         "test": {"accuracy": 0.6, "cost_per_query": 0.0011}},
        {"name": "ghost", "description": "(recovered)", "code": ""},   # no code -> dropped
    ]}
    trace = {"records": [
        {"question": {"text": "3*5?"}, "gold": {"text": "15"},
         "answer": {"text": "15 apples"}, "score": 0.0, "cost": 0.001,
         "error": None, "calls": []}]}
    search = rebuild_search(result, trace_for=lambda name: trace)

    assert [c.name for c in search.archive] == ["H"]
    carried = search.archive[0]
    assert carried.dev.accuracy == 0.5 and carried.dev.cached_input_frac == 0.2
    assert carried.test.accuracy == 0.6              # kept, so ranking won't re-buy it
    # the per-example failures feed the next round's archive summary
    assert "15 apples" in summarize_archive(search.archive)


def test_ranking_keeps_carried_test_scores_instead_of_rebuying_them(catalog):
    from flowopt.optimizer import Search, optimize

    cfg = load_config("gsm8k", ["designer.rounds=0", "designer.research=false"])
    carried = Candidate("A", "", GOOD["code"], dev=SplitScore("A", 0.9, 0.001),
                        test=SplitScore("A", 0.77, 0.001))
    fresh = Candidate("B", "", GOOD["code"] + "#", dev=SplitScore("B", 0.95, 0.010))
    search = Search(archive=[carried, fresh])

    out = optimize(cfg, benchmark_fixture(), evaluator(cfg, catalog),
                   log=lambda *a: None, search=search)
    by_name = {c.name: c for c in out.finalists}
    assert by_name["A"].test.accuracy == 0.77        # the paid-for score, untouched
    assert by_name["B"].test.accuracy == 1.0         # freshly evaluated


def test_operator_guidance_reaches_the_round_prompt(cfg):
    benchmark = benchmark_fixture()
    nudge = "STOP EXPLORING VOTING VARIANTS; DECOMPOSE THE MULTI-PART QUESTIONS"
    with_nudge = _round_prompt(cfg, benchmark, 1, "", guidance=nudge)
    assert nudge in with_nudge and "OPERATOR GUIDANCE" in with_nudge
    assert "OPERATOR GUIDANCE" not in _round_prompt(cfg, benchmark, 1, "")


def test_a_continued_round_one_extends_the_frontier_it_inherited(cfg):
    # round 1 WITH an archive must get the improve goal, not "design a diverse set"
    benchmark = benchmark_fixture()
    archive = [Candidate("H", "", "carried-code", dev=SplitScore("H", 0.8, 0.001))]
    prompt = _round_prompt(cfg, benchmark, 1, summarize_archive(archive))
    assert "carried-code" in prompt and "Pareto frontier" in prompt
    assert "DIVERSE" not in prompt


def test_research_notes_are_handed_to_the_design_agent(cfg):
    benchmark = benchmark_fixture()
    marker = "SELF-CONSISTENCY OF 3 IS THE KNOWN WIN HERE"
    with_notes = _round_prompt(cfg, benchmark, 1, "", research_notes=marker)
    assert marker in with_notes                       # the designer sees what research found
    assert marker not in _round_prompt(cfg, benchmark, 1, "")   # and nothing leaks when skipped


def test_research_collect_notes_reads_the_file_or_returns_empty(tmp_path):
    from flowopt.research import _collect_notes
    assert _collect_notes(tmp_path) == ""             # no file yet -> "", not an error
    (tmp_path / "research_notes.md").write_text("# findings\nuse a cheap→opus cascade")
    assert "cascade" in _collect_notes(tmp_path)


def test_the_archive_summary_gives_the_full_set_and_marks_the_frontier():
    archive = [Candidate("H", "", "cheap-code", dev=SplitScore("H", 0.70, 0.0002)),
               Candidate("S", "", "good-code", dev=SplitScore("S", 0.90, 0.0010)),
               Candidate("D", "", "dud-code", dev=SplitScore("D", 0.60, 0.0050))]
    summary = summarize_archive(archive)
    # the whole set is handed over now, dominated ones included, so a new design
    # can borrow from any of them
    assert all(code in summary for code in ("cheap-code", "good-code", "dud-code"))
    # the frontier is marked so the agent knows which points it has to beat; D is
    # dominated by both H (cheaper and more accurate) and S, so only H and S carry it
    assert summary.count("[ON FRONTIER]") == 2


def test_the_archive_summary_caps_dominated_but_keeps_all_frontier():
    # BEST is cheapest AND most accurate, so it dominates every dud and is the
    # sole frontier; the 15 duds are all off-frontier.
    best = Candidate("BEST", "", "code-best", dev=SplitScore("BEST", 0.99, 0.0001))
    duds = [Candidate(f"d{i}", "", f"DUD_{i}_", dev=SplitScore(f"d{i}", 0.50, 0.0010))
            for i in range(15)]
    summary = summarize_archive([best] + duds, dominated_shown=3)

    assert summary.count("[ON FRONTIER]") == 1 and "code-best" in summary
    shown = sum(1 for i in range(15) if f"DUD_{i}_" in summary)
    assert shown == 3                                    # only the cap's worth of dominated
    assert "DUD_14_" in summary and "DUD_0_" not in summary   # most recent are the ones kept
    assert "12 more dominated" in summary                # and the drop is stated, not silent


def test_the_archive_summary_surfaces_a_candidates_dev_failures():
    records = [
        {"question": "2+2?", "gold": "4", "answer": "4", "score": 1.0, "error": None},
        {"question": "3*5?", "gold": "15", "answer": "15 apples", "score": 0.0, "error": None},
        {"question": "9-1?", "gold": "8", "answer": "", "score": 0.0,
         "error": "RuntimeError: workflow exceeded its model-call budget"},
    ]
    summary = summarize_archive(
        [Candidate("W", "", "code", dev=SplitScore("W", 0.33, 0.001, records=records))])
    assert "Lost points on" in summary
    assert "15 apples" in summary          # the wrong answer, so the agent sees the format bug
    assert "model-call budget" in summary  # the error, so it sees the budget blow-up
    assert "2+2" not in summary            # the example it got right is not fed back as noise


# ---- a broken grader must not read as a run of zeros ------------------------
def test_a_grader_that_cannot_read_the_data_fails_before_the_search(cfg):
    """A grader that raises is caught per example and scored 0.0, so a broken
    grader looks exactly like a workflow that answers wrongly. This once cost a
    full search — 9 candidates over 120 examples, every score 0 — to discover
    that the dataset lacked the `doc` field its grader reads."""
    def needs_doc(prediction, item):
        return float(item["doc"]["ok"])          # the field the data hasn't got

    grader = Grader(kind="custom", grade_fn=needs_doc)
    with pytest.raises(ValueError) as raised:
        analysis.check_grader(grader, {"question": "q", "answer": "a"})
    message = str(raised.value)
    assert "cannot read this dataset" in message
    assert "KeyError" in message
    assert "['answer', 'question']" in message   # says what the data actually has


def test_a_working_grader_passes_the_check():
    analysis.check_grader(Grader(kind="exact"), {"question": "q", "answer": "a"})
    analysis.check_grader(Grader(kind="numeric"), {"question": "q", "answer": "42"})


def test_the_judge_is_not_probed(monkeypatch):
    # probing a judge costs an API call, and it fails by scoring low, not raising
    judge = Grader(kind="llm_judge", client=None)
    analysis.check_grader(judge, {"question": "q", "answer": "a"})     # must not call out


# ---- n_examples has to mean the same thing for loaded and generated data -----
def test_n_examples_caps_a_loaded_dataset():
    """Asking for 40 against a 200-row benchmark used to run all 200 — and the
    cost estimate, which sizes itself from n_examples, understated it 5x."""
    from flowopt import dataset as datasets

    data = [{"question": f"q{i}", "answer": str(i)} for i in range(200)]
    taken = datasets.take(data, 40, log=lambda *a: None)
    assert len(taken) == 40
    assert all(item in data for item in taken)


def test_taking_is_deterministic_so_two_runs_are_comparable():
    from flowopt import dataset as datasets

    data = [{"question": f"q{i}", "answer": str(i)} for i in range(200)]
    first = datasets.take(data, 40, log=lambda *a: None)
    second = datasets.take(data, 40, log=lambda *a: None)
    assert [d["question"] for d in first] == [d["question"] for d in second]


def test_asking_for_more_than_exists_keeps_everything():
    from flowopt import dataset as datasets

    data = [{"question": f"q{i}", "answer": str(i)} for i in range(10)]
    assert len(datasets.take(data, 40, log=lambda *a: None)) == 10
    assert len(datasets.take(data, 0, log=lambda *a: None)) == 10      # 0 means all


# ---- a task can forbid the tools its workflows may use ----------------------
def test_a_forbidden_tool_is_rejected_at_the_call_site(cfg, catalog):
    """A closed-book task sets tools=[]; a candidate that still calls web_search
    must fail, not quietly answer another way, or the benchmark isn't closed-book."""
    from flowopt.runtime import CallMeter

    meter = CallMeter(FakeClient(catalog), catalog.default, 24, 120_000, allowed_tools=[])
    with pytest.raises(RuntimeError) as raised:
        meter.call_model("q", tools=["web_search"])
    assert "not allowed" in str(raised.value)


def test_an_allowed_tool_passes(cfg, catalog):
    from flowopt.runtime import CallMeter

    meter = CallMeter(FakeClient(catalog), catalog.default, 24, 120_000,
                      allowed_tools=["web_search"])
    meter.call_model("q", tools=["web_search"])              # must not raise
    assert meter.calls == 1


def test_code_execution_is_rejected_with_a_pointer(cfg, catalog):
    """There is no server-side code tool via OpenRouter. A workflow asking for it
    gets told so by name, not a bare 'not allowed'."""
    from flowopt.runtime import CallMeter

    meter = CallMeter(FakeClient(catalog), catalog.default, 24, 120_000,
                      allowed_tools=["web_search", "web_fetch"])
    with pytest.raises(RuntimeError) as raised:
        meter.call_model("q", tools=["code_execution"])
    assert "no code_execution" in str(raised.value)
    meter.call_model("q", tools=["web_search", "web_fetch"])    # the real menu is fine


def test_the_meter_prefers_openrouters_billed_cost(catalog):
    # OpenRouter reports what it actually charged; the catalog math is only the
    # fallback for clients that don't attach it
    from flowopt.runtime import CallMeter

    class BilledClient(FakeClient):
        def call(self, model, prompt, **kw):
            response = super().call(model, prompt, **kw)
            response.cost = 0.5
            return response

    meter = CallMeter(BilledClient(catalog), catalog.default, 24, 120_000)
    meter.call_model("add 1 and 2")
    assert meter.cost == 0.5


def test_no_allowlist_means_no_restriction(cfg, catalog):
    from flowopt.runtime import CallMeter

    meter = CallMeter(FakeClient(catalog), catalog.default, 24, 120_000)  # allowed_tools=None
    meter.call_model("q", tools=["web_search"])              # anything goes
    assert meter.calls == 1


def test_the_evaluator_enforces_the_configs_tools(catalog):
    cfg = load_config("gsm8k", ["runtime.tools=[]"])
    searcher = {"name": "searcher", "code": (
        "def solve(q, call_model):\n"
        "    return call_model(q, tools=['web_search'])\n")}
    result = Evaluator(FakeClient(catalog), Grader(kind="numeric"), cfg.runtime).run(searcher, DATA)
    assert result.accuracy == 0.0                            # every example rejected
    assert all("not allowed" in e for e in result.errors)
