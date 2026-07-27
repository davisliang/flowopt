#!/usr/bin/env python3
"""The flowopt UI: a stdlib HTTP server over the run directory.

Serves one static page plus a small JSON API. Every read comes off disk, and
every search runs in its own subprocess, so the server keeps no state: restart
it mid-search and the page picks up exactly where it was.

Usage:
    uv run flowopt-ui                 # http://127.0.0.1:8770
    uv run flowopt-ui --port 9000

Binds to localhost by default. Starting a search spends real money, so anything
that can reach this port can spend it.
"""
import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import webbrowser
from dataclasses import asdict
from typing import Optional
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import analysis, catalog as feeds, costs, paths, runstore
from ..client import PLUGIN_DEFS, TOOL_DEFS
from ..config import load_config, load_resolved
from ..paths import ROOT
from ..session import Session

STATIC_INDEX = Path(__file__).parent / "static" / "index.html"

_MODEL_ID = re.compile(r"[A-Za-z0-9][\w.:/-]*$")


def _model_id(raw) -> str:
    """One OpenRouter model id from the form, validated to id-shaped characters."""
    raw = str(raw).strip()
    if not _MODEL_ID.match(raw):
        raise ValueError(raw)
    return raw


def _model_list(raw) -> str:
    """The form's model pool as an OmegaConf dotlist value.

    Accepts a JSON array (the POST body) or a comma-joined string (GET estimate
    params). Ids are quoted so slashes and dots survive the dotlist grammar.
    """
    items = raw if isinstance(raw, list) else str(raw).split(",")
    ids = [_model_id(item) for item in items if str(item).strip()]
    if not ids:
        raise ValueError("empty model list")
    return "[" + ",".join(f"'{i}'" for i in ids) + "]"


def _flag(raw) -> str:
    """A boolean the form sends as bool or text, normalized for the dotlist."""
    value = str(raw).strip().lower()
    if value in ("true", "1", "yes", "on"):
        return "true"
    if value in ("false", "0", "no", "off"):
        return "false"
    raise ValueError(raw)


def _plugin_list(raw) -> str:
    """The form's request-plugin selection, validated against the registry.

    Unlike models this may legitimately be empty — [] turns every plugin off.
    """
    items = raw if isinstance(raw, list) else [p for p in str(raw).split(",") if p.strip()]
    names = [str(p).strip() for p in items]
    for name in names:
        if name not in PLUGIN_DEFS:
            raise ValueError(name)
    return "[" + ",".join(f"'{n}'" for n in names) + "]"


# Overrides the New Search form can set, and how to read each one. Anything not
# on this list is rejected: values reach OmegaConf, and the form is not a shell.
FORM_FIELDS = {
    "models": _model_list,
    "designer.model": _model_id,
    "designer.research": _flag,
    "designer.research_model": _model_id,
    "designer.max_turns": int,
    "designer.research_max_turns": int,
    "call.plugins": _plugin_list,
    "designer.rounds": int,
    "data.n_examples": int,
    "data.n_train": int,
    "data.n_dev": int,
    "data.n_test": int,
    "runtime.concurrency": int,
    "runtime.max_model_calls": int,
    "runtime.max_tokens": int,
    "report.max_cost_per_query": float,
    "report.min_accuracy": float,
}

# The server-side tools a workflow may be granted — sourced from the client's
# registry so the form can never offer one the API doesn't know how to send.
ALLOWED_WORKFLOW_TOOLS = sorted(TOOL_DEFS)


def _dotlist(overrides: dict) -> tuple[list, str]:
    """Turn the form's overrides into OmegaConf dotlist entries, or say why not.

    The one gate every form-supplied setting passes through, so starting a run,
    estimating one, and probing one all reject exactly the same inputs — three
    hand-rolled copies of this loop had already drifted apart on unknown keys.

    Args:
        overrides: Config overrides keyed by dotted path, as the form sends them.
            None or {} is fine. A blank value means "use the default" and is
            skipped, since that is what the form submits for an untouched field.

    Returns:
        `(entries, "")` on success, or `([], reason)` when a key is not in
        FORM_FIELDS or a value doesn't parse as that field's type.
    """
    entries = []
    for key, raw in (overrides or {}).items():
        if key not in FORM_FIELDS:
            return [], f"unknown setting: {key}"
        if raw is None or raw == "":
            continue
        try:
            entries.append(f"{key}={FORM_FIELDS[key](raw)}")
        except (TypeError, ValueError):
            return [], f"bad value for {key}: {raw!r}"
    return entries, ""


def parse_dataset(text: str) -> tuple[list, str]:
    """Read an uploaded dataset into examples.

    Accepts JSONL or a JSON array, with the answer under "answer", "target" or
    "gold" — the three spellings the exports around here use.

    Args:
        text: The uploaded file's contents.

    Returns:
        `(examples, "")` on success, or `([], reason)` if it can't be read.
    """
    text = (text or "").strip()
    if not text:
        return [], "the dataset is empty"

    rows = []
    if text.startswith("["):
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as error:
            return [], f"not valid JSON: {error}"
    else:
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                return [], f"line {number} is not valid JSON: {error}"

    examples = []
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            return [], f"row {number} is not an object"
        question = row.get("question") or row.get("input") or row.get("prompt")
        answer = row.get("answer", row.get("target", row.get("gold")))
        if question is None or answer is None:
            return [], (f"row {number} needs a question and an answer — saw keys "
                        f"{sorted(row)[:6]}")
        examples.append({**row, "question": str(question), "answer": answer})

    if len(examples) < 2:
        return [], "at least 2 examples are needed, to split dev from test"
    return examples, ""


def start_run(task: str, overrides: dict, prompt: str = "", dataset_text: str = "",
              tools: list = None, extra_skills: list = None,
              working_skills: Optional[bool] = None) -> dict:
    """Create a run directory and launch the pipeline against it.

    Args:
        task: A task config name. Must be one of `runstore.list_tasks()` — it
            names a file path, so an unknown value is rejected rather than
            resolved. Ignored when `prompt` is given.
        overrides: Config overrides keyed by dotted path. Only keys in
            FORM_FIELDS are accepted, each coerced to that field's type.
        prompt: A free-text task description. Starts an ad-hoc search instead of
            using a task file; the analyzer infers the grading rule from it.
        dataset_text: An uploaded JSONL or JSON array of examples. Optional — a
            free-text task with no data generates its own.
        tools: Server-side tools workflows may use, a subset of
            ALLOWED_WORKFLOW_TOOLS. None leaves the task's config default; a list
            (including []) overrides it, so [] forbids all tools.
        extra_skills: Skills to stage the design agent with IN ADDITION to the
            config's own set, from `runstore.list_skills()` names. Additions
            only, deliberately: the round prompt drives the agent through the
            core skills by name, so a form that could drop one would produce
            agents instructed to use a skill they don't have — every candidate
            malformed, the whole round paid for and wasted. Removing core
            skills is an expert move, and config is where experts go.
        working_skills: Whether the agent keeps a run-scoped skills directory it
            writes and re-reads across rounds. None leaves the config default.

    Returns:
        `{"ok": True, "run_id": ...}`, or `{"ok": False, "error": ...}`.
    """
    if tools is not None:
        bad = [x for x in tools if x not in ALLOWED_WORKFLOW_TOOLS]
        if bad:
            return {"ok": False, "error": f"unknown tool(s): {', '.join(bad)}"}
    if extra_skills is not None:
        known = {s["name"] for s in runstore.list_skills()}
        bad = [x for x in extra_skills if x not in known]
        if bad:
            return {"ok": False, "error": f"unknown skill(s): {', '.join(bad)}"}
    freetext = bool(prompt and prompt.strip())
    if not freetext and task not in runstore.list_tasks():
        return {"ok": False, "error": f"unknown task: {task}"}

    dotlist, reason = _dotlist(overrides)
    if reason:
        return {"ok": False, "error": reason}

    examples = []
    if dataset_text:
        examples, reason = parse_dataset(dataset_text)
        if reason:
            return {"ok": False, "error": f"dataset: {reason}"}

    try:
        if freetext:
            # Nothing from the form names a file: the task is built in memory,
            # and any uploaded data is written inside the run's own directory.
            cfg = load_config("", dotlist)
            cfg.task.name = "custom"
            cfg.task.seed_prompt = prompt.strip()
        else:
            cfg = load_config(task, dotlist)
    except Exception as error:
        return {"ok": False, "error": f"config: {error}"}

    if tools is not None:
        cfg.runtime.tools = list(tools)
    if extra_skills:
        staged = list(cfg.designer.skills)
        cfg.designer.skills = staged + [s for s in extra_skills if s not in staged]
    if working_skills is not None:
        cfg.designer.working_skills = bool(working_skills)

    status = runstore.create_run(cfg.task.name, cfg)
    if examples:
        data_file = runstore.run_dir(status.run_id) / "dataset.jsonl"
        data_file.write_text("".join(json.dumps(e) + "\n" for e in examples))
        cfg.task.dataset = str(data_file)
    if examples or tools is not None or extra_skills or working_skills is not None:
        runstore.write_config(status.run_id, cfg)
    _spawn_runner(status.run_id)
    return {"ok": True, "run_id": status.run_id}


def _spawn_runner(run_id: str) -> None:
    """Launch the pipeline subprocess against an already-created run directory.

    Args:
        run_id: The run to execute.
    """
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "flowopt.dashboard.runner", run_id],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        start_new_session=True,       # its own process group, so Stop takes the agent too
    )
    runstore.update_status(run_id, pid=process.pid)


def open_run_dir(run_id: str) -> dict:
    """Open one run's directory in the OS file browser.

    The server runs on the user's own machine, so "show me the run's files" can
    actually mean Finder / Explorer. Only a validated run id resolves — the
    endpoint takes no path of its own, so nothing outside `runs/` can be named.

    Args:
        run_id: The run whose directory to open.

    Returns:
        `{"ok": True, "path": ...}`, or `{"ok": False, "error": ...}`.
    """
    status = runstore.read_status(run_id)
    if status is None:
        return {"ok": False, "error": "unknown run"}
    directory = runstore.run_dir(run_id)
    opener = {"darwin": ["open"], "win32": ["explorer"]}.get(sys.platform, ["xdg-open"])
    try:
        subprocess.Popen(opener + [str(directory)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError as error:
        return {"ok": False, "error": f"could not open the folder: {error}"}
    return {"ok": True, "path": str(directory)}


def continue_run(run_id: str, rounds, guidance: str = "") -> dict:
    """Resume a finished search: more design rounds on top of everything it learned.

    A fresh run directory is seeded from the source run — its resolved config
    (rounds overridden), its saved benchmark (the SAME dev/test splits, so
    scores stay comparable), its result (the archive the new rounds design
    against), its traces (the per-example failures that say where the old
    designs break), its research notes, and its working skills. The optional
    guidance is the operator's nudge, handed to every design round.

    Args:
        run_id: The run to continue.
        rounds: How many more design rounds to run, 1-10.
        guidance: Free text telling the next rounds where to focus. Optional.

    Returns:
        `{"ok": True, "run_id": <the new run>}`, or `{"ok": False, "error": ...}`.
        Runs from before continuation support lack a saved benchmark and are
        refused — their dataset cannot be reconstructed exactly.
    """
    status = runstore.read_status(run_id)
    if status is None:
        return {"ok": False, "error": "unknown run"}
    if status.state == "running":
        return {"ok": False, "error": "the run is still going — stop it or let it finish first"}
    try:
        rounds = int(rounds)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"bad rounds value: {rounds!r}"}
    if not 1 <= rounds <= 10:
        return {"ok": False, "error": "rounds must be between 1 and 10"}

    source_dir = runstore.run_dir(run_id)
    missing = [name for name in ("config.yaml", "result.json", "benchmark.json")
               if not (source_dir / name).exists()]
    if missing:
        reason = f"this run can't be continued — missing {', '.join(missing)}"
        if "benchmark.json" in missing:
            reason += ("; runs from before continuation support didn't save their "
                       "dataset, and a generated one can't be rebuilt identically")
        return {"ok": False, "error": reason}

    cfg = load_resolved(source_dir / "config.yaml")
    cfg.designer.rounds = rounds
    status = runstore.create_run(status.task, cfg)
    new_dir = runstore.run_dir(status.run_id)
    shutil.copy(source_dir / "benchmark.json", new_dir / "benchmark.json")
    shutil.copy(source_dir / "result.json", new_dir / "source_result.json")
    if (source_dir / "research_notes.md").exists():
        shutil.copy(source_dir / "research_notes.md", new_dir / "research_notes.md")
    for folder in ("traces", "skills"):        # carried candidates stay inspectable;
        if (source_dir / folder).exists():     # the agent's self-built skills carry over
            shutil.copytree(source_dir / folder, new_dir / folder)
    (new_dir / "continue.json").write_text(json.dumps(
        {"source": run_id, "rounds": rounds, "guidance": str(guidance or "")[:4000]}))
    _spawn_runner(status.run_id)
    return {"ok": True, "run_id": status.run_id}


def compare_runs() -> dict:
    """Gather every scored candidate across every run, for the comparison chart.

    One point per candidate per run, so searches on the same benchmark — or on
    different ones — can be read on the same accuracy/cost axes.

    Returns:
        `{"points": [...], "baselines": {task: {...}}}`. Each point carries the
        run it came from, its task, the split its numbers are from, accuracy,
        cost, whether it was on that run's frontier, and its description.
    """
    points, tasks = [], set()
    for status in runstore.list_runs():
        detail_events = runstore.read_events(status.run_id)
        result = runstore.read_result(status.run_id) or {}
        frontier = set(result.get("frontier", []))
        seen = {}
        for event in detail_events:
            if event.get("event") == "candidate":
                seen[event["name"]] = {"dev": (event["dev_accuracy"], event["dev_cost"]),
                                       "description": event.get("description", "")}
            elif event.get("event") == "test_scored" and event["name"] in seen:
                seen[event["name"]]["test"] = (event["test_accuracy"], event["test_cost"])
        for name, entry in seen.items():
            split = "test" if "test" in entry else "dev"
            accuracy, cost = entry.get("test") or entry["dev"]
            points.append({"run_id": status.run_id, "task": status.task, "name": name,
                           "split": split, "accuracy": accuracy, "cost": cost,
                           "frontier": name in frontier,
                           "description": entry.get("description", "")})
        if seen:
            tasks.add(status.task)

    return {"points": points,
            "baselines": {task: runstore.baselines_for(task) for task in sorted(tasks)}}


def estimate_cost(task: str, overrides: dict, freetext: bool = False,
                  has_dataset: bool = False) -> dict:
    """Estimate what the form's current settings would cost to run.

    Args:
        task: The selected task config name, or "" for a free-text task.
        overrides: The form's settings, same keys as `start_run` accepts.
        freetext: Whether this is a described-in-prose task.
        has_dataset: Whether a dataset was uploaded, so none is generated.

    Returns:
        The Estimate as a dict, or `{"error": ...}` if the settings don't load —
        the same rejections `start_run` would give, surfaced before spending.
    """
    dotlist, reason = _dotlist(overrides)
    if reason:
        return {"error": reason}
    if not freetext and task not in runstore.list_tasks():
        return {"error": f"unknown task: {task}"}

    try:
        cfg = load_config("" if freetext else task, dotlist)
    except Exception as error:
        return {"error": f"config: {error}"}

    history = costs.observed([(s.task, runstore.read_events(s.run_id))
                              for s in runstore.list_runs()])
    generates = not (has_dataset or (not freetext and bool(cfg.task.dataset)))
    shape = _dataset_shape(cfg)
    guess = costs.estimate(cfg, history, generates_data=generates,
                           judged=None if not freetext else False,
                           available=shape["rows"] if shape else None, allocated=shape)
    return _estimate_dict(guess)


def probe_and_estimate(task: str, overrides: dict) -> dict:
    """Measure this task with a few real calls, then estimate from that.

    Costs a few cents and takes seconds. Unlike history, it works on a task
    nobody has ever run, and it measures the two things that actually drive cost:
    how many tokens a call on this task takes, and whether the cheap model can do
    the work at all.

    Args:
        task: A task config name. Free-text tasks cannot be probed — there are no
            examples until the dataset is generated.
        overrides: The form's settings.

    Returns:
        The estimate, with a "probe" block describing the measurement, or
        `{"error": ...}`.
    """
    base = estimate_cost(task, overrides)
    if base.get("error"):
        return base

    dotlist, _ = _dotlist(overrides)       # estimate_cost above already validated
    cfg = load_config(task, dotlist)
    if not cfg.task.dataset:
        return {**base, "probe": {"skipped": "this task generates its own examples, "
                                  "so there is nothing to probe until it runs"}}

    session = Session.from_config(cfg)
    try:
        benchmark = analysis.build_benchmark(cfg, session.client, log=lambda *a: None)
    except Exception as error:
        return {"error": f"could not load the task: {error}"}

    measured = costs.run_probe(session.client, benchmark.grader, benchmark.dev, n=3)
    if not measured.n:
        return {**base, "probe": {"skipped": "every probe call failed — the API may be "
                                  "busy; the estimate below is from defaults"}}

    history = costs.observed([(s.task, runstore.read_events(s.run_id))
                              for s in runstore.list_runs()])
    shape = _dataset_shape(cfg)
    guess = costs.estimate(cfg, history, generates_data=False, probe=measured,
                           available=shape["rows"] if shape else None, allocated=shape)
    return _estimate_dict(guess, probe={
        "n": measured.n, "model": measured.model,
        "input_tokens": measured.input_tokens,
        "output_tokens": measured.output_tokens,
        "accuracy": measured.accuracy, "cost": measured.cost})


def compare_examples(run_id: str, split: str = "dev", limit: int = 200) -> dict:
    """Line every workflow's answer to the same example up side by side.

    A per-candidate accuracy says which workflow won; it doesn't say where they
    differed, which is the thing worth reading. Aligning answers by example shows
    exactly which questions separate a cheap workflow from an expensive one — and
    whether the expensive one is right for a reason or just lucky.

    Args:
        run_id: The run to read.
        split: "dev" or "test".
        limit: Most rows to return.

    Returns:
        `{"candidates": [names], "rows": [...], "split": ...}`. Each row carries
        the question, the gold answer, one cell per candidate, and `spread` —
        the gap between the best and worst score on that example, so the rows
        that discriminate can be found first.
    """
    status = runstore.read_status(run_id)
    if status is None:
        return {"error": "not_found"}

    names = [e["name"] for e in runstore.read_events(run_id)
             if e.get("event") == "candidate"]
    traces = [(name, runstore.read_trace(run_id, name, split)) for name in names]
    traces = [(name, trace) for name, trace in traces if trace]
    if not traces:
        return {"candidates": [], "rows": [], "split": split,
                "note": f"no {split} traces recorded for this run"}

    # Align on the question text. Candidates are scored over the same split in the
    # same order, but matching on content survives a reordering.
    rows: dict[str, dict] = {}
    for name, trace in traces:
        for record in trace["records"]:
            question = record["question"]["text"]
            row = rows.setdefault(question, {
                "question": question, "gold": record["gold"]["text"], "cells": {}})
            row["cells"][name] = {
                "answer": record["answer"]["text"], "clipped": record["answer"]["clipped"],
                "score": record["score"], "error": record["error"],
                "cost": record["cost"], "calls": len(record["calls"])}

    candidates = [name for name, _ in traces]
    out = []
    for row in rows.values():
        scores = [c["score"] for c in row["cells"].values()]
        out.append({**row,
                    "cells": [row["cells"].get(name) for name in candidates],
                    "spread": (max(scores) - min(scores)) if scores else 0.0})
    # Most-disagreed first: a row every workflow got right teaches nothing.
    out.sort(key=lambda r: (-r["spread"], r["question"]))
    return {"candidates": candidates, "rows": out[:limit], "split": split,
            "n_rows": len(out)}


def run_detail(run_id: str, log_lines: int = 400) -> dict:
    """Assemble everything the detail pane shows for one run.

    Args:
        run_id: The run to describe.
        log_lines: How many trailing log lines to include. The UI asks for more
            when the reader opens the full log.

    Returns:
        Its status, milestones, candidates (merged from live events and the saved
        result so a running search and a finished one render the same way), the
        research notes, the log tail, and the frontier. `{"error": "not_found"}`
        if unknown.
    """
    status = runstore.read_status(run_id)
    if status is None:
        return {"error": "not_found"}

    events = runstore.read_events(run_id)
    result = runstore.read_result(run_id)

    # While running, candidates come from the event stream; once finished, the
    # saved result carries the code and the test scores too.
    candidates: dict[str, dict] = {}
    for event in events:
        if event.get("event") == "candidate":
            candidates[event["name"]] = {
                "name": event["name"], "description": event.get("description", ""),
                "round": event.get("round"),
                "dev": {"accuracy": event["dev_accuracy"], "cost_per_query": event["dev_cost"],
                        "cached_input_frac": event.get("cached_input_frac", 0.0),
                        "errors": event.get("errors", [])},
                "test": None, "code": "",
            }
        elif event.get("event") == "test_scored" and event["name"] in candidates:
            candidates[event["name"]]["test"] = {
                "accuracy": event["test_accuracy"], "cost_per_query": event["test_cost"]}
    for saved in (result or {}).get("candidates", []):
        merged = candidates.setdefault(saved["name"], {"name": saved["name"]})
        merged.update({k: saved[k] for k in ("description", "code") if k in saved})
        for split in ("dev", "test"):
            if saved.get(split):
                merged[split] = saved[split]

    analyzed = next((e for e in events if e.get("event") == "analyzed"), {})
    return {
        "status": _status_dict(status),
        "analysis": {"check": analyzed.get("check", ""),
                     "description": analyzed.get("description", ""),
                     "judge_status": analyzed.get("judge_status", ""),
                     "rubric": analyzed.get("rubric", ""),
                     "answer_examples": analyzed.get("answer_examples", []),
                     "train_sample": analyzed.get("train_sample", []),
                     "dev_sample": analyzed.get("dev_sample", []),
                     "test_sample": analyzed.get("test_sample", [])},
        "candidates": list(candidates.values()),
        "frontier": (result or {}).get("frontier", []),
        "research": runstore.read_research(run_id),
        "log": runstore.read_log(run_id, max_lines=log_lines),
        "config": runstore.read_config_text(run_id),
        "events": events,
    }


def model_directory(refresh: bool = False) -> dict:
    """Every model OpenRouter serves, annotated for the form's picker.

    OpenRouter supplies existence, price and context; Artificial Analysis (when
    ARTIFICIAL_ANALYSIS_API_KEY is set) the capability and speed measurements.
    Both are read through the same disk cache the search itself uses, so what
    the picker shows is what a run would bill against.

    Args:
        refresh: True refetches both feeds even if the cache is fresh.

    Returns:
        {"models": [...], "default_pool": [...], "default_designer": id,
         "aa_available": bool} — models sorted cheapest first, each with id,
        name, price_in/out (USD per 1M), context_length, thinks, and its AA
        summary or None.
    """
    openrouter = feeds.openrouter_models(refresh)
    aa = feeds.aa_models(refresh)
    base = load_config()
    models = []
    for model_id, record in openrouter.items():
        pricing = record.get("pricing", {})
        per_m = lambda key: (float(pricing[key]) * 1_000_000
                             if pricing.get(key) not in (None, "") else None)
        models.append({
            "id": model_id,
            "name": record.get("name"),
            "price_in": per_m("prompt"),
            "price_out": per_m("completion"),
            "context_length": record.get("context_length"),
            "thinks": "reasoning" in (record.get("supported_parameters") or []),
            "aa": feeds.aa_summary(model_id, aa),
        })
    models.sort(key=lambda m: (m["price_in"] or 0, m["price_out"] or 0, m["id"]))
    return {"models": models,
            "default_pool": list(base.models),
            "default_designer": base.designer.model,
            "aa_available": bool(aa)}


def _dataset_shape(cfg) -> Optional[dict]:
    """Count a task's dataset rows and any fixed partition labels.

    The run scores what the data allows, not what was asked: `n_examples` caps
    a loaded pool, and an allocated benchmark's blank dev/test take a labeled
    partition WHOLE. An estimate that ignores either can be out by the ratio —
    a request for 40 against 200 rows understated fivefold, and bbeh's
    417-example holdout would cost ~5x an estimate assuming the fraction split.

    Args:
        cfg: The run config.

    Returns:
        `{"rows", "train", "val", "test"}` counts, or None if the task
        generates its own examples or the file cannot be read.
    """
    if not cfg.task.dataset:
        return None
    shape = {"rows": 0, "train": 0, "val": 0, "test": 0}
    try:
        with open(paths.resolve(cfg.task.dataset)) as f:
            for line in f:
                if not line.strip():
                    continue
                shape["rows"] += 1
                part = json.loads(line).get("split")
                part = "test" if part == "holdout" else part
                if part in ("train", "val", "test"):
                    shape[part] += 1
    except (OSError, ValueError):
        return None
    return shape


def _estimate_dict(guess: costs.Estimate, **extra) -> dict:
    """Render an Estimate as the JSON shape both estimate endpoints return.

    Args:
        guess: The estimate to convert.
        **extra: Additional fields to include — e.g. the probe description.

    Returns:
        The estimate's fields as a plain dict, plus `extra`.
    """
    return {"low": guess.low, "expected": guess.expected, "high": guess.high,
            "breakdown": guess.breakdown, "assumptions": guess.assumptions,
            "based_on_runs": guess.based_on_runs, **extra}


def _status_dict(status) -> dict:
    """Render a RunStatus as JSON-safe fields.

    Args:
        status: The RunStatus to convert.

    Returns:
        Its fields as a plain dict.
    """
    return asdict(status)


class Handler(BaseHTTPRequestHandler):
    """Routes the UI's requests. One instance per request, as BaseHTTPRequestHandler wants."""

    def _json(self, obj, code: int = 200) -> None:
        """Send a JSON response.

        Args:
            obj: Any JSON-serializable object.
            code: HTTP status code.
        """
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str) -> None:
        """Send an HTML response.

        Args:
            html: The page source.
        """
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> dict:
        """Read and parse the request's JSON body.

        Returns:
            The parsed object, or {} if the body is empty or malformed.
        """
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def do_GET(self):
        """Serve the page, the task list, the run list, or one run's detail."""
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                if not STATIC_INDEX.exists():
                    return self._html(f"<h1>500</h1><p>missing {STATIC_INDEX}</p>")
                return self._html(STATIC_INDEX.read_text(encoding="utf-8"))
            if path == "/api/tasks":
                base = load_config()
                # The config's own skills are marked required: the round prompt
                # drives the agent through them by name, so the form offers
                # additions only — dropping one is a config-level decision.
                core = set(base.designer.skills)
                skills = [{**s, "required": s["name"] in core}
                          for s in runstore.list_skills()]
                return self._json({"tasks": runstore.list_tasks(),
                                   "benchmarks": runstore.list_benchmarks(),
                                   "fields": sorted(FORM_FIELDS),
                                   "workflow_tools": ALLOWED_WORKFLOW_TOOLS,
                                   "default_tools": list(base.runtime.tools),
                                   "plugins": sorted(PLUGIN_DEFS),
                                   "default_plugins": list(base.call.plugins),
                                   "default_research": bool(base.designer.research),
                                   "default_research_model": (base.designer.research_model
                                                              or base.designer.model),
                                   "skills": skills,
                                   "default_working_skills": bool(base.designer.working_skills)})
            if path == "/api/models":
                params = parse_qs(urlparse(self.path).query)
                refresh = (params.get("refresh") or ["0"])[0] == "1"
                return self._json(model_directory(refresh))
            if path == "/api/compare":
                return self._json(compare_runs())
            if path == "/api/estimate":
                params = parse_qs(urlparse(self.path).query)
                overrides = {k: v[0] for k, v in params.items()
                             if k in FORM_FIELDS and v and v[0] != ""}
                return self._json(estimate_cost(
                    (params.get("task") or [""])[0],
                    overrides,
                    freetext=(params.get("freetext") or ["0"])[0] == "1",
                    has_dataset=(params.get("has_dataset") or ["0"])[0] == "1"))
            if path == "/api/runs":
                return self._json({"runs": [_status_dict(s) for s in runstore.list_runs()]})
            if path.startswith("/api/answers/"):
                run_id = path[len("/api/answers/"):]
                if not runstore.is_valid_run_id(run_id):
                    return self._json({"error": "bad run id"}, code=400)
                params = parse_qs(urlparse(self.path).query)
                split = (params.get("split") or ["dev"])[0]
                if split not in ("dev", "test"):
                    return self._json({"error": "split must be dev or test"}, code=400)
                result = compare_examples(run_id, split)
                return self._json(result, code=404 if result.get("error") else 200)
            if path.startswith("/api/trace/"):
                run_id = path[len("/api/trace/"):]
                if not runstore.is_valid_run_id(run_id):
                    return self._json({"error": "bad run id"}, code=400)
                params = parse_qs(urlparse(self.path).query)
                name = (params.get("name") or [""])[0]
                split = (params.get("split") or ["dev"])[0]
                if split not in ("dev", "test"):
                    return self._json({"error": "split must be dev or test"}, code=400)
                trace = runstore.read_trace(run_id, name, split)
                return self._json(trace or {"error": "no trace recorded"},
                                  code=200 if trace else 404)
            if path.startswith("/api/run/"):
                run_id = path[len("/api/run/"):]
                if not runstore.is_valid_run_id(run_id):
                    return self._json({"error": "bad run id"}, code=400)
                params = parse_qs(urlparse(self.path).query)
                lines = min(int((params.get("log_lines") or ["400"])[0] or 400), 20000)
                detail = run_detail(run_id, log_lines=lines)
                return self._json(detail, code=404 if detail.get("error") else 200)
            self.send_error(404, "Not Found")
        except Exception as error:
            self._json({"error": str(error)}, code=500)

    def do_POST(self):
        """Start a search, or stop a running one."""
        path = urlparse(self.path).path
        try:
            if path == "/api/probe":
                body = self._body()
                return self._json(probe_and_estimate(body.get("task", ""),
                                                     body.get("overrides", {})))
            if path == "/api/runs":
                body = self._body()
                result = start_run(body.get("task", ""), body.get("overrides", {}),
                                   prompt=body.get("prompt", ""),
                                   dataset_text=body.get("dataset", ""),
                                   tools=body.get("tools"),
                                   extra_skills=body.get("extra_skills"),
                                   working_skills=body.get("working_skills"))
                return self._json(result, code=200 if result.get("ok") else 400)
            if path.startswith("/api/run/") and path.endswith("/stop"):
                run_id = path[len("/api/run/"):-len("/stop")]
                if not runstore.is_valid_run_id(run_id):
                    return self._json({"error": "bad run id"}, code=400)
                result = runstore.stop_run(run_id)
                return self._json(result, code=200 if result.get("ok") else 400)
            if path.startswith("/api/run/") and path.endswith("/continue"):
                run_id = path[len("/api/run/"):-len("/continue")]
                if not runstore.is_valid_run_id(run_id):
                    return self._json({"error": "bad run id"}, code=400)
                body = self._body()
                result = continue_run(run_id, body.get("rounds", 2),
                                      guidance=body.get("guidance", ""))
                return self._json(result, code=200 if result.get("ok") else 400)
            if path.startswith("/api/run/") and path.endswith("/open"):
                run_id = path[len("/api/run/"):-len("/open")]
                if not runstore.is_valid_run_id(run_id):
                    return self._json({"error": "bad run id"}, code=400)
                result = open_run_dir(run_id)
                return self._json(result, code=200 if result.get("ok") else 400)
            if path.startswith("/api/run/") and path.endswith("/delete"):
                run_id = path[len("/api/run/"):-len("/delete")]
                if not runstore.is_valid_run_id(run_id):
                    return self._json({"error": "bad run id"}, code=400)
                result = runstore.delete_run(run_id)
                return self._json(result, code=200 if result.get("ok") else 400)
            self.send_error(404, "Not Found")
        except Exception as error:
            self._json({"error": str(error)}, code=500)

    def log_message(self, fmt, *args):
        """Silence the default per-request logging."""


def main() -> None:
    """Serve the UI until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--open", action="store_true", help="open a browser on start")
    args = parser.parse_args()

    # Reap finished searches automatically. Without this a completed run stays in
    # the process table as a zombie, which still answers `kill(pid, 0)` — so the
    # run list would report a finished run as still running. The server never
    # calls wait() itself, so nothing here depends on collecting exit statuses.
    if hasattr(signal, "SIGCHLD"):
        signal.signal(signal.SIGCHLD, signal.SIG_IGN)

    runstore.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    url = f"http://{args.host}:{args.port}"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"flowopt UI → {url}")
    print("Starting a search spends real money. Ctrl-C to stop the server.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
