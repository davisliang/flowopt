"""The UI's logic, without a browser or an API key.

Covers what the run store promises (state survives on disk, ids can't escape
`runs/`) and what the server rejects, since the form's values reach OmegaConf and
starting a search spends money. Run with `uv run pytest`.
"""
import json
import sys

import pytest

from flowopt import costs, runstore
from flowopt.config import load_config, load_resolved
from flowopt.dashboard import server


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Point the run store at a temporary directory for the duration of a test."""
    monkeypatch.setattr(runstore, "RUNS_DIR", tmp_path / "runs")
    runstore.RUNS_DIR.mkdir(parents=True)
    return runstore.RUNS_DIR


@pytest.fixture
def a_run(runs_dir):
    """Create one run directory and return its status."""
    return runstore.create_run("gsm8k", load_config("gsm8k", ["designer.rounds=1"]))


# ---- run ids are filesystem paths, so they are validated ---------------------
@pytest.mark.parametrize("bad", [
    "../../etc/passwd", "a/b", "..", "", "x" * 200, "has space", "semi;colon",
])
def test_unsafe_run_ids_are_rejected(bad):
    assert not runstore.is_valid_run_id(bad)
    with pytest.raises(ValueError):
        runstore.run_dir(bad)


def test_ordinary_run_ids_are_accepted():
    assert runstore.is_valid_run_id("gsm8k-20260720-143012")
    assert runstore.is_valid_run_id("ifeval-20260720-143012-2")


# ---- a run's state lives on disk --------------------------------------------
def test_creating_a_run_writes_config_and_status(a_run, runs_dir):
    directory = runs_dir / a_run.run_id
    assert (directory / "config.yaml").exists()
    assert (directory / "status.json").exists()
    assert a_run.task == "gsm8k"
    assert a_run.rounds == 1
    assert a_run.state == "running"


def test_status_survives_a_round_trip(a_run):
    runstore.update_status(a_run.run_id, phase="designing", round=2, n_candidates=3)
    reloaded = runstore.read_status(a_run.run_id)
    assert (reloaded.phase, reloaded.round, reloaded.n_candidates) == ("designing", 2, 3)


def test_events_append_in_order(a_run):
    runstore.append_event(a_run.run_id, {"event": "analyzing"})
    runstore.append_event(a_run.run_id, {"event": "candidate", "name": "H"})
    events = runstore.read_events(a_run.run_id)
    assert [e["event"] for e in events] == ["analyzing", "candidate"]
    assert all("t" in e for e in events)          # every event is timestamped


def test_a_half_written_event_line_is_skipped(a_run):
    # the UI polls while the runner is appending, so it can read a torn last line
    runstore.append_event(a_run.run_id, {"event": "analyzing"})
    with open(runstore.run_dir(a_run.run_id) / "events.jsonl", "a") as f:
        f.write('{"event": "candi')
    assert [e["event"] for e in runstore.read_events(a_run.run_id)] == ["analyzing"]


def test_a_run_whose_process_died_is_not_still_running(a_run):
    # a pid that cannot exist: the machine restarted, or the process was killed
    runstore.update_status(a_run.run_id, pid=2 ** 30)
    listed = runstore.list_runs()
    assert listed[0].state == "failed"
    assert "no longer running" in listed[0].error


def test_reading_an_unknown_run_returns_none(runs_dir):
    assert runstore.read_status("nope-20260101-000000") is None
    assert runstore.read_events("nope-20260101-000000") == []
    assert runstore.read_result("nope-20260101-000000") is None


# ---- what the server accepts from the form ----------------------------------
def test_an_unknown_task_cannot_name_a_file(runs_dir):
    result = server.start_run("../../etc/passwd", {})
    assert result["ok"] is False and "unknown task" in result["error"]


def test_only_listed_settings_can_be_overridden(runs_dir):
    result = server.start_run("gsm8k", {"task.grader": "/etc/passwd"})
    assert result["ok"] is False and "unknown setting" in result["error"]


def test_a_setting_that_is_not_a_number_is_rejected(runs_dir):
    result = server.start_run("gsm8k", {"designer.rounds": "; rm -rf /"})
    assert result["ok"] is False and "bad value" in result["error"]


def test_blank_settings_fall_back_to_the_config_default(runs_dir, monkeypatch):
    # the form submits "" for a field the user left alone
    started = {}
    real_create_run = runstore.create_run

    def spy(task, cfg):
        started["cfg"] = cfg
        return real_create_run(task, cfg)

    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 4242})())
    monkeypatch.setattr(server.runstore, "create_run", spy)

    result = server.start_run("gsm8k", {"designer.rounds": "", "data.n_examples": 12})
    assert result["ok"] is True
    assert started["cfg"].data.n_examples == 12                      # the value given
    assert started["cfg"].designer.rounds == load_config("gsm8k").designer.rounds  # the default


# ---- what the detail endpoint returns ---------------------------------------
def test_detail_merges_live_events_into_candidates(a_run):
    runstore.append_event(a_run.run_id, {
        "event": "candidate", "round": 1, "name": "H", "description": "one call",
        "dev_accuracy": 0.8, "dev_cost": 0.001, "cached_input_frac": 0.0, "errors": []})
    runstore.append_event(a_run.run_id, {
        "event": "test_scored", "name": "H", "test_accuracy": 0.75, "test_cost": 0.0011})

    detail = server.run_detail(a_run.run_id)
    assert detail["status"]["run_id"] == a_run.run_id
    candidate = detail["candidates"][0]
    assert candidate["dev"]["accuracy"] == 0.8
    assert candidate["test"]["accuracy"] == 0.75
    assert candidate["code"] == ""          # code only lands when the search finishes


def test_detail_prefers_the_saved_result_once_it_exists(a_run):
    runstore.append_event(a_run.run_id, {
        "event": "candidate", "round": 1, "name": "H", "description": "",
        "dev_accuracy": 0.8, "dev_cost": 0.001, "cached_input_frac": 0.0, "errors": []})
    (runstore.run_dir(a_run.run_id) / "result.json").write_text(json.dumps({
        "frontier": ["H"],
        "candidates": [{"name": "H", "description": "one call", "code": "def solve(): ...",
                        "dev": {"accuracy": 0.8, "cost_per_query": 0.001},
                        "test": {"accuracy": 0.9, "cost_per_query": 0.0012}}]}))

    detail = server.run_detail(a_run.run_id)
    candidate = detail["candidates"][0]
    assert candidate["code"] == "def solve(): ..."
    assert candidate["test"]["accuracy"] == 0.9
    assert detail["frontier"] == ["H"]


def test_detail_of_an_unknown_run_is_not_found(runs_dir):
    assert server.run_detail("nope-20260101-000000") == {"error": "not_found"}


def test_research_notes_round_trip_and_surface_in_the_detail(a_run):
    assert runstore.read_research(a_run.run_id) == ""        # none until the phase runs
    runstore.write_research(a_run.run_id, "# findings\nroute easy inputs to haiku")
    assert "haiku" in runstore.read_research(a_run.run_id)
    assert "haiku" in server.run_detail(a_run.run_id)["research"]


def test_researching_is_a_phase(a_run):
    # the UI renders PHASES as pills in order; research sits between analyze and design
    assert runstore.PHASES.index("researching") == runstore.PHASES.index("analyzing") + 1
    assert runstore.PHASES.index("researching") < runstore.PHASES.index("designing")


def test_deleting_a_run_is_permanent_but_never_touches_a_live_one(a_run):
    import os
    # "running" with a live process is refused — stop it first
    runstore.update_status(a_run.run_id, pid=os.getpid())
    assert "still going" in runstore.delete_run(a_run.run_id)["error"]
    assert runstore.run_dir(a_run.run_id).exists()
    # finished runs delete, directory and all
    runstore.update_status(a_run.run_id, state="done")
    assert runstore.delete_run(a_run.run_id)["ok"] is True
    assert not runstore.run_dir(a_run.run_id).exists()
    assert "unknown run" in runstore.delete_run(a_run.run_id)["error"]


def test_deleting_cannot_reach_outside_runs():
    # the id is a filesystem path, so the validation that guards reads guards this too
    import pytest as _pytest
    with _pytest.raises(ValueError):
        runstore.delete_run("../../etc")


def test_stopping_a_finished_run_says_so(a_run):
    runstore.update_status(a_run.run_id, state="done")
    assert runstore.stop_run(a_run.run_id)["ok"] is False


def test_a_zombie_process_does_not_count_as_running(a_run):
    """A finished-but-unreaped child still answers `kill(pid, 0)`.

    Taken as alive, a crashed run would sit in the list as "running" forever —
    which is exactly what a real run did before `_process_alive` learned to reap.
    """
    import os
    import subprocess as sp
    import time

    child = sp.Popen([sys.executable, "-c", "pass"])
    deadline = time.time() + 5
    while time.time() < deadline:       # wait for it to actually become a zombie
        state = sp.run(["ps", "-o", "stat=", "-p", str(child.pid)],
                       capture_output=True, text=True).stdout.strip()
        if state.startswith("Z"):
            break
        time.sleep(0.05)
    else:
        pytest.skip("could not produce a zombie on this platform")

    os.kill(child.pid, 0)               # the zombie is still signallable...
    runstore.update_status(a_run.run_id, pid=child.pid)
    assert runstore.list_runs()[0].state == "failed"   # ...but it is not running


# ---- free-text tasks and uploaded datasets ----------------------------------
def test_an_uploaded_jsonl_is_read(runs_dir):
    text = '{"question": "a", "answer": "1"}\n{"question": "b", "answer": "2"}\n'
    examples, reason = server.parse_dataset(text)
    assert reason == ""
    assert [e["question"] for e in examples] == ["a", "b"]


def test_a_json_array_and_alternate_key_names_are_read(runs_dir):
    # exports around here spell the gold "target" or "gold", and the input "prompt"
    text = '[{"prompt": "a", "target": "1"}, {"input": "b", "gold": "2"}]'
    examples, reason = server.parse_dataset(text)
    assert reason == "" and len(examples) == 2
    assert examples[0]["answer"] == "1" and examples[1]["question"] == "b"


def test_extra_columns_survive_for_a_custom_grader(runs_dir):
    text = '{"question": "a", "answer": "1", "doc": {"k": 2}}\n{"question": "b", "answer": "2"}\n'
    examples, _ = server.parse_dataset(text)
    assert examples[0]["doc"] == {"k": 2}


@pytest.mark.parametrize("text,expected", [
    ("", "empty"),
    ("not json", "not valid JSON"),
    ('{"question": "a"}\n{"question": "b", "answer": "x"}', "needs a question and an answer"),
    ('{"question": "a", "answer": "1"}', "at least 2 examples"),
])
def test_a_bad_dataset_is_refused_with_a_reason(runs_dir, text, expected):
    examples, reason = server.parse_dataset(text)
    assert examples == [] and expected in reason


def test_a_freetext_task_needs_no_task_file(runs_dir, monkeypatch):
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 4242})())
    result = server.start_run("", {}, prompt="Classify sentiment as positive or negative.")
    assert result["ok"] is True
    cfg = load_resolved(runstore.run_dir(result["run_id"]) / "config.yaml")
    assert cfg.task.name == "custom"
    assert "Classify sentiment" in cfg.task.seed_prompt


def test_an_uploaded_dataset_lands_in_the_run_and_is_pointed_at(runs_dir, monkeypatch):
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 4242})())
    result = server.start_run("", {}, prompt="Label it.",
                              dataset_text='{"question": "a", "answer": "1"}\n'
                                           '{"question": "b", "answer": "2"}\n')
    assert result["ok"] is True
    directory = runstore.run_dir(result["run_id"])
    assert (directory / "dataset.jsonl").exists()
    cfg = load_resolved(directory / "config.yaml")
    assert cfg.task.dataset == str(directory / "dataset.jsonl")


def test_a_freetext_run_still_rejects_unlisted_settings(runs_dir):
    result = server.start_run("", {"task.grader": "/etc/passwd"}, prompt="Label it.")
    assert result["ok"] is False and "unknown setting" in result["error"]


def test_opening_a_runs_folder_names_only_that_directory(a_run, monkeypatch):
    spawned = []
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda args, **k: spawned.append(args) or type("P", (), {"pid": 1})())
    assert "unknown run" in server.open_run_dir("nope-20260101-000000")["error"]
    result = server.open_run_dir(a_run.run_id)
    assert result["ok"] is True
    # the opener is handed exactly the run's directory, nothing else from outside
    assert spawned[0][-1] == str(runstore.run_dir(a_run.run_id))
    assert result["path"] == str(runstore.run_dir(a_run.run_id))


# ---- continuing a finished search -------------------------------------------
def test_continuing_refuses_what_it_cannot_resume(a_run):
    # still running
    assert "still going" in server.continue_run(a_run.run_id, 2)["error"]
    assert "unknown run" in server.continue_run("nope-20260101-000000", 2)["error"]
    runstore.update_status(a_run.run_id, state="done")
    assert "rounds must be" in server.continue_run(a_run.run_id, 0)["error"]
    assert "bad rounds" in server.continue_run(a_run.run_id, "lots")["error"]
    # finished, but nothing saved to resume from — the reason names the files
    result = server.continue_run(a_run.run_id, 2)
    assert result["ok"] is False and "benchmark.json" in result["error"]


def test_continuing_seeds_a_new_run_from_the_source(a_run, monkeypatch):
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 4242})())
    source = runstore.run_dir(a_run.run_id)
    runstore.update_status(a_run.run_id, state="done")
    (source / "result.json").write_text(json.dumps({"candidates": [
        {"name": "H", "description": "", "code": "def solve(q, m): return '5'",
         "dev": {"accuracy": 0.5, "cost_per_query": 0.001}}]}))
    (source / "benchmark.json").write_text(json.dumps({
        "analysis": {"description": "Add.", "check_type": "numeric",
                     "judge_rubric": "", "answer_examples": []},
        "dev": [{"question": "q", "answer": "1"}],
        "test": [{"question": "q2", "answer": "2"}],
        "judge_status": "", "grader": {"kind": "numeric", "task": "", "rubric": ""}}))
    (source / "research_notes.md").write_text("# use a cascade")
    (source / "traces").mkdir()
    (source / "traces" / "sample.json").write_text("{}")

    result = server.continue_run(a_run.run_id, 3, guidance="focus on decomposition")
    assert result["ok"] is True
    new_dir = runstore.run_dir(result["run_id"])
    marker = json.loads((new_dir / "continue.json").read_text())
    assert marker["source"] == a_run.run_id
    assert marker["guidance"] == "focus on decomposition"
    # the new run carries the benchmark, archive, notes and traces...
    assert (new_dir / "benchmark.json").exists()
    assert (new_dir / "source_result.json").exists()
    assert "cascade" in (new_dir / "research_notes.md").read_text()
    assert (new_dir / "traces" / "sample.json").exists()
    # ...and runs the asked-for number of extra rounds against the same task
    cfg = load_resolved(new_dir / "config.yaml")
    assert cfg.designer.rounds == 3
    assert runstore.read_status(result["run_id"]).task == "gsm8k"


# ---- benchmarks and the comparison view -------------------------------------
def test_benchmarks_are_listed_with_their_metadata():
    found = {b["name"]: b for b in runstore.list_benchmarks()}
    if not found:
        pytest.skip("benchmarks/ not imported in this checkout")
    assert "ifeval" in found and "arc_agi_2" in found
    assert found["ifeval"]["baselines"]["haiku"] == pytest.approx(0.8478, abs=1e-4)
    # partition counts surface for the picker; unallocated benchmarks read zero
    assert found["gpqa_diamond_gen"]["splits"]["test"] == 25
    assert found["gpqa_diamond_gen"]["splits"]["train"] > 0
    assert found["arc_agi_2"]["splits"]["test"] == 0
    # the code tasks cannot be graded here, and say so rather than scoring wrongly
    for name in ("humaneval_plus_gen", "mbpp_plus"):
        if name in found:
            assert found[name]["supported"] is False and found[name]["note"]


def test_compare_puts_every_run_on_the_same_axes(a_run):
    runstore.append_event(a_run.run_id, {
        "event": "candidate", "round": 1, "name": "H", "description": "one call",
        "dev_accuracy": 0.8, "dev_cost": 0.001, "cached_input_frac": 0.0, "errors": []})
    runstore.append_event(a_run.run_id, {
        "event": "test_scored", "name": "H", "test_accuracy": 0.7, "test_cost": 0.0012})

    compared = server.compare_runs()
    point = compared["points"][0]
    assert point["run_id"] == a_run.run_id and point["task"] == "gsm8k"
    assert point["split"] == "test"            # test scores win where they exist
    assert point["accuracy"] == 0.7


# ---- estimating a run before paying for it ----------------------------------
def test_an_estimate_says_which_figures_were_measured(runs_dir):
    result = server.estimate_cost("gsm8k", {"designer.rounds": "2", "data.n_examples": "40"})
    assert result["expected"] > 0
    assert result["low"] < result["expected"] < result["high"]
    assert "design agent" in result["breakdown"]
    # with no history every figure must be labelled a default, not passed off as known
    assert result["based_on_runs"] == 0
    assert all("measured" not in a for a in result["assumptions"] if "past run" in a)


def test_past_runs_replace_the_defaults(a_run):
    for round_num in (1, 2):
        runstore.append_event(a_run.run_id, {"event": "round_start", "round": round_num,
                                             "rounds": 2})
        runstore.append_event(a_run.run_id, {"event": "agent_cost", "round": round_num,
                                             "usd": 0.40, "turns": 20})
        runstore.append_event(a_run.run_id, {
            "event": "candidate", "round": round_num, "name": f"H{round_num}",
            "description": "", "dev_accuracy": 0.5, "dev_cost": 0.002,
            "cached_input_frac": 0.0, "errors": []})

    history = costs.observed([('gsm8k', runstore.read_events(a_run.run_id))])
    mine = history["tasks"]["gsm8k"]
    assert mine["agent_cost_per_round"] == pytest.approx(0.40)
    assert mine["cost_per_query"] == pytest.approx(0.002)
    assert mine["candidates_per_round"] == pytest.approx(1.0)

    result = server.estimate_cost("gsm8k", {"designer.rounds": "2"})
    assert result["based_on_runs"] == 1
    assert any("measured" in a for a in result["assumptions"])
    # the measured agent cost is used, not the much larger default
    assert result["breakdown"]["design agent"] == pytest.approx(0.80)


def test_more_rounds_and_examples_cost_more(runs_dir):
    small = server.estimate_cost("gsm8k", {"designer.rounds": "1", "data.n_examples": "20"})
    big = server.estimate_cost("gsm8k", {"designer.rounds": "4", "data.n_examples": "200"})
    assert big["expected"] > small["expected"] * 3


def test_a_supplied_dataset_removes_the_generation_cost(runs_dir):
    generated = server.estimate_cost("", {"data.n_examples": "100"}, freetext=True)
    uploaded = server.estimate_cost("", {"data.n_examples": "100"}, freetext=True,
                                    has_dataset=True)
    assert "generate examples" in generated["breakdown"]
    assert "generate examples" not in uploaded["breakdown"]
    assert uploaded["expected"] < generated["expected"]


def test_an_estimate_refuses_the_same_things_a_run_would(runs_dir):
    assert "unknown task" in server.estimate_cost("../etc/passwd", {})["error"]
    assert "bad value" in server.estimate_cost("gsm8k", {"designer.rounds": "; rm -rf /"})["error"]
    # an unknown key used to be silently skipped here while start_run rejected it
    assert "unknown setting" in server.estimate_cost("gsm8k", {"task.grader": "x"})["error"]


# ---- a stopped or failed run must not lose what it already paid for ---------
def test_partial_results_are_written_when_a_run_is_cut_short(a_run, monkeypatch):
    """A stopped ARC run once discarded six scored candidates and ~$260 of work,
    because results were only written after the very last step."""
    from flowopt.dashboard import runner
    from flowopt.optimizer import Candidate, Search
    from flowopt.runtime import SplitScore

    search = Search(archive=[
        Candidate("H", "one call", "def solve(q, m): pass", dev=SplitScore("H", 0.4, 0.01)),
        Candidate("S", "two calls", "def solve(q, m): pass", dev=SplitScore("S", 0.6, 0.02)),
    ])
    cfg = load_resolved(runstore.run_dir(a_run.run_id) / "config.yaml")
    kept = runner._write_result(a_run.run_id, cfg, search)

    assert kept == 2
    saved = runstore.read_result(a_run.run_id)
    assert [c["name"] for c in saved["candidates"]] == ["H", "S"]
    assert saved["candidates"][0]["dev"]["accuracy"] == 0.4


def test_a_run_whose_config_cannot_load_is_marked_failed(a_run):
    """`cfg` used to be bound inside the try, so a config that failed to load
    crashed the failure handler itself and left the run "running" forever."""
    from flowopt.dashboard import runner

    (runstore.run_dir(a_run.run_id) / "config.yaml").write_text("{ not yaml [")
    assert runner.main(a_run.run_id) == 1
    status = runstore.read_status(a_run.run_id)
    assert status.state == "failed"
    assert status.error                     # the real message, not an UnboundLocalError


def test_writing_results_for_an_empty_search_is_a_no_op(a_run):
    from flowopt.dashboard import runner
    from flowopt.optimizer import Search

    cfg = load_resolved(runstore.run_dir(a_run.run_id) / "config.yaml")
    assert runner._write_result(a_run.run_id, cfg, Search()) == 0
    assert runner._write_result(a_run.run_id, cfg, None) == 0
    assert runstore.read_result(a_run.run_id) is None


def test_the_agent_cost_line_matches_what_the_proposer_prints():
    """The design agent's spend crosses a process boundary as a log line, so the
    two sides are only connected by this format. Nothing else would notice if
    one drifted — the estimator would just silently keep using its default."""
    from flowopt.designer import AGENT_COST

    # built exactly as proposer.py builds it
    cost, turns = 1.2345, 37
    line = f"[agent cost: ${cost:.4f} over {turns} turns]"

    found = AGENT_COST.search(line)
    assert found, f"designer cannot parse the line proposer prints: {line!r}"
    assert float(found.group(1)) == pytest.approx(cost, abs=1e-4)
    assert int(found.group(2)) == turns


def test_the_agent_cost_line_is_found_amid_other_output():
    from flowopt.designer import AGENT_COST
    assert AGENT_COST.search("  [tool] Bash") is None
    assert AGENT_COST.search("[agent finished: success]") is None
    assert AGENT_COST.search("[agent cost: $0.0500 over 3 turns]").group(1) == "0.0500"


def test_cost_per_query_is_not_borrowed_between_tasks():
    """ARC measured $0.74 a query against ifeval's $0.0018. Pooling those made
    every task estimate the same number and ARC's wrong by ~30x."""
    history = costs.observed([
        ("arc_agi_2", [{"event": "round_start", "round": 1},
                       {"event": "candidate", "dev_cost": 0.74, "dev_accuracy": 0.5}]),
        ("ifeval", [{"event": "round_start", "round": 1},
                    {"event": "candidate", "dev_cost": 0.0018, "dev_accuracy": 0.5}]),
    ])
    assert history["tasks"]["arc_agi_2"]["cost_per_query"] == pytest.approx(0.74)
    assert history["tasks"]["ifeval"]["cost_per_query"] == pytest.approx(0.0018)

    arc = costs.estimate(load_config("arc_agi_2", ["data.n_examples=40"]), history)
    ifeval = costs.estimate(load_config("ifeval", ["data.n_examples=40"]), history)
    assert arc.expected > ifeval.expected * 20        # the task dominates, as it must

    # a task with no history of its own uses the default, not the polluted pool
    gsm8k = costs.estimate(load_config("gsm8k", ["data.n_examples=40"]), history)
    assert gsm8k.expected < arc.expected / 10
    assert any("task-specific" in a for a in gsm8k.assumptions)


def test_the_designer_is_told_the_cost_target():
    """The budget used to filter only the final recommendation, so the agent could
    spend a whole search designing workflows nobody would pick."""
    from flowopt.designer import _round_prompt

    cfg = load_config("gsm8k", ["report.max_cost_per_query=0.004"])
    from flowopt.analysis import Benchmark, TaskAnalysis
    from flowopt.grading import Grader
    benchmark = Benchmark(
        analysis=TaskAnalysis(description="Add numbers.", check_type="numeric",
                              judge_rubric="", answer_examples=["5"]),
        grader=Grader(kind="numeric"), dev=[{"question": "q", "answer": "1"}],
        test=[{"question": "q", "answer": "1"}])
    prompt = _round_prompt(cfg, benchmark, 1, "")
    assert "0.00400" in prompt and "Cost target" in prompt


# ---- probing beats guessing --------------------------------------------------
class ProbeClient:
    """A client that reports fixed token usage, so probe maths can be checked."""

    def __init__(self, catalog, input_tokens=1000, output_tokens=500, answer="x"):
        self.catalog = catalog
        self.usage = {"input": input_tokens, "output": output_tokens,
                      "cache_write": 0, "cache_read": 0}
        self.answer = answer
        self.calls = 0

    def call(self, model, prompt, system=None, tools=None, effort=None, schema=None):
        from flowopt.client import ApiResponse
        self.calls += 1
        return ApiResponse(text=json.dumps({"answer": self.answer}), usage=dict(self.usage))


def test_a_probe_measures_tokens_and_whether_the_cheap_model_can_do_it():
    from flowopt.grading import Grader
    from flowopt.models import ModelCatalog

    cfg = load_config("gsm8k")
    catalog = ModelCatalog.from_config(cfg)
    data = [{"question": "q1", "answer": "x"}, {"question": "q2", "answer": "x"},
            {"question": "q3", "answer": "no"}]

    probe = costs.run_probe(cfg, ProbeClient(catalog), Grader(kind="exact"), data, n=3)
    assert probe.n == 3
    assert probe.input_tokens == 1000 and probe.output_tokens == 500
    assert probe.accuracy == pytest.approx(2 / 3)     # two of three golds are "x"
    assert probe.cost > 0


def test_a_probe_that_cannot_reach_the_api_degrades_instead_of_breaking():
    from flowopt.grading import Grader
    from flowopt.models import ModelCatalog

    class Broken(ProbeClient):
        def call(self, *a, **k):
            raise RuntimeError("overloaded")

    cfg = load_config("gsm8k")
    probe = costs.run_probe(cfg, Broken(ModelCatalog.from_config(cfg)), Grader(kind="exact"),
                            [{"question": "q", "answer": "a"}], n=3)
    assert probe.n == 0          # the caller falls back to defaults rather than failing


def test_a_failing_cheap_model_predicts_an_expensive_search():
    """The probe's real job: if the cheap tier scores ~0 the designer escalates,
    which is why ARC cost ~60x its own baseline while ifeval stayed cheap."""
    from flowopt.models import ModelCatalog

    catalog = ModelCatalog.from_config(load_config("gsm8k"))
    same_tokens = dict(input_tokens=5000, output_tokens=1000, output_tokens_high=1000, n=3)
    hopeless = costs.Probe(accuracy=0.0, **same_tokens)
    capable = costs.Probe(accuracy=0.9, **same_tokens)

    _, expensive, _, why_expensive = costs.per_query_from_probe(catalog, hopeless)
    _, cheap, _, why_cheap = costs.per_query_from_probe(catalog, capable)

    assert expensive > cheap * 10          # same tokens, wildly different forecast
    assert "escalate" in why_expensive and "cheap workflows are viable" in why_cheap


def test_a_probe_overrides_history_and_says_so():
    cfg = load_config("gsm8k", ["data.n_examples=40"])
    history = {"tasks": {"gsm8k": {"cost_per_query": 0.5}}, "pooled": {}, "n_runs": 9}
    probe = costs.Probe(input_tokens=100, output_tokens=50, output_tokens_high=50,
                        accuracy=1.0, n=3, model="claude-haiku-4-5")

    with_probe = costs.estimate(cfg, history, probe=probe)
    without = costs.estimate(cfg, history)
    assert with_probe.expected < without.expected      # the probe says it is cheap
    assert any("derived from that probe" in a for a in with_probe.assumptions)


def test_the_estimate_sizes_itself_from_the_dataset_not_the_request():
    """A run scores min(n_examples, dataset size); an estimate that assumes
    n_examples is out by the ratio between them."""
    cfg = load_config("gsm8k", ["data.n_examples=200"])
    asked = costs.estimate(cfg, {})
    capped = costs.estimate(cfg, {}, available=40)

    # the example-dependent work scales with the real dev size (train is carved
    # out first); the design agent, fixed per round, does not
    n_train = int(cfg.data.n_train)
    dev_of = lambda n: int((n - n_train) * float(cfg.data.dev_fraction))  # noqa: E731
    assert capped.breakdown["score on dev"] == pytest.approx(
        asked.breakdown["score on dev"] * dev_of(40) / dev_of(200), rel=0.01)
    assert capped.breakdown["design agent"] == asked.breakdown["design agent"]
    assert capped.expected < asked.expected
    assert any("fewer than the 200 requested" in a for a in capped.assumptions)


def test_the_estimate_knows_blank_sizes_take_a_partition_whole(runs_dir):
    """bbeh's 417-example holdout would cost ~5x an estimate that assumed the
    fraction split — blank dev/test on an allocated benchmark run the whole
    partition, and the estimate must say so."""
    result = server.estimate_cost("bbeh_gen", {})
    assert any("allocation" in a and "417 test" in a for a in result["assumptions"]), \
        result["assumptions"]


# ---- comparing answers across workflows, example by example -----------------
def _write_trace(run_id, name, split, records):
    """Write a trace file directly, in the shape compare_examples reads."""
    traces = runstore.run_dir(run_id) / "traces"
    traces.mkdir(exist_ok=True)
    payload = {"candidate": name, "split": split, "records": [
        {"question": {"text": q, "clipped": False}, "gold": {"text": g, "clipped": False},
         "answer": {"text": a, "clipped": False}, "score": s, "cost": 0.001,
         "error": None, "calls": []} for q, g, a, s in records]}
    (traces / runstore.trace_name(name, split)).write_text(json.dumps(payload))


def test_answers_line_up_by_example_across_workflows(a_run):
    for name in ("H", "S"):
        runstore.append_event(a_run.run_id, {"event": "candidate", "name": name,
            "description": "", "dev_accuracy": 0.5, "dev_cost": 0.001,
            "cached_input_frac": 0.0, "errors": []})
    _write_trace(a_run.run_id, "H", "dev",
        [("2+2?", "4", "4", 1.0), ("hard one", "42", "41", 0.0)])
    _write_trace(a_run.run_id, "S", "dev",
        [("2+2?", "4", "4", 1.0), ("hard one", "42", "42", 1.0)])

    result = server.compare_examples(a_run.run_id, "dev")
    assert result["candidates"] == ["H", "S"]
    assert result["n_rows"] == 2
    # the row they disagree on comes first
    top = result["rows"][0]
    assert top["question"] == "hard one"
    assert top["spread"] == 1.0
    assert [c["answer"] for c in top["cells"]] == ["41", "42"]
    assert [c["score"] for c in top["cells"]] == [0.0, 1.0]


def test_a_candidate_missing_from_a_split_is_a_gap_not_a_crash(a_run):
    for name in ("H", "S"):
        runstore.append_event(a_run.run_id, {"event": "candidate", "name": name,
            "description": "", "dev_accuracy": 0.5, "dev_cost": 0.001,
            "cached_input_frac": 0.0, "errors": []})
    _write_trace(a_run.run_id, "H", "dev", [("q", "a", "a", 1.0)])
    # S was never scored on dev — only H has a trace

    result = server.compare_examples(a_run.run_id, "dev")
    assert result["candidates"] == ["H"]        # only the candidates that have traces
    assert result["rows"][0]["cells"][0]["answer"] == "a"


def test_comparing_answers_needs_traces(a_run):
    runstore.append_event(a_run.run_id, {"event": "candidate", "name": "H",
        "description": "", "dev_accuracy": 0.5, "dev_cost": 0.001,
        "cached_input_frac": 0.0, "errors": []})
    result = server.compare_examples(a_run.run_id, "dev")
    assert result["rows"] == [] and "no dev traces" in result["note"]


def test_skills_are_listed_for_the_picker():
    names = {s["name"] for s in runstore.list_skills()}
    assert "workflow-design" in names and "workflow-eval" in names
    # the harness-managed skills are not offered — they ride their own toggles
    assert "workflow-skills" not in names and "workflow-research" not in names
    assert all(s["description"] for s in runstore.list_skills())


def test_extra_skills_add_to_the_core_set_never_replace_it(runs_dir, monkeypatch, tmp_path):
    """The round prompt drives the agent through the core skills by name, so the
    form can only ADD skills — a form that could drop workflow-design would
    produce a full-price round of malformed candidates."""
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 4242})())
    # a made-up skill and a harness-managed one are both refused — they name
    # directories, and the form is not a shell
    assert "unknown skill" in server.start_run("gsm8k", {}, extra_skills=["telepathy"])["error"]
    assert "unknown skill" in server.start_run(
        "gsm8k", {}, extra_skills=["workflow-skills"])["error"]

    # a skill the user dropped into skills/ is appended to the core set
    fake_skills = tmp_path / "skillset"
    (fake_skills / "my-domain").mkdir(parents=True)
    (fake_skills / "my-domain" / "SKILL.md").write_text(
        "---\nname: my-domain\ndescription: domain notes\n---\nnotes")
    monkeypatch.setattr(runstore, "SKILLS_DIR", fake_skills)

    core = list(load_config("gsm8k").designer.skills)
    result = server.start_run("gsm8k", {}, extra_skills=["my-domain"], working_skills=True)
    assert result["ok"] is True
    cfg = load_resolved(runstore.run_dir(result["run_id"]) / "config.yaml")
    assert list(cfg.designer.skills) == core + ["my-domain"]   # added — core intact
    assert cfg.designer.working_skills is True

    # untouched, both keep the config default
    result = server.start_run("gsm8k", {})
    cfg = load_resolved(runstore.run_dir(result["run_id"]) / "config.yaml")
    assert list(cfg.designer.skills) == core
    assert cfg.designer.working_skills is False


def test_the_tools_selection_persists_and_is_validated(runs_dir, monkeypatch):
    monkeypatch.setattr(server.subprocess, "Popen",
                        lambda *a, **k: type("P", (), {"pid": 4242})())
    # a bogus tool is refused
    assert "unknown tool" in server.start_run("gsm8k", {}, tools=["telepathy"])["error"]
    assert set(server.ALLOWED_WORKFLOW_TOOLS) == {"code_execution", "web_search", "web_fetch"}
    # a closed-book choice is written to the run's config
    result = server.start_run("gsm8k", {}, tools=[])
    assert result["ok"] is True
    cfg = load_resolved(runstore.run_dir(result["run_id"]) / "config.yaml")
    assert list(cfg.runtime.tools) == []
