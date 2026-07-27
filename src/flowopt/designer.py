"""Step 2 — the design agent that writes candidate workflows.

Each round runs a Claude Agent SDK session in a scratch directory, driven by the
three skills under `skills/`. On round 1 it designs a diverse initial set; on
later rounds it is shown every workflow tried so far — code, dev scores, and the
examples each got wrong — and asked to design new ones that extend the
accuracy/cost frontier.

The agent runs in a SEPARATE PROCESS (`flowopt.proposer`). A notebook
kernel that has called `nest_asyncio.apply()` monkeypatches `asyncio` for the
whole session and breaks `asyncio.run`, even from a worker thread; a subprocess
is immune.
"""
import json
import os
import re
import pathlib
import shutil
import subprocess
import sys
import tempfile

from omegaconf import OmegaConf

from . import prompts
from .pareto import DEV, pareto_front
from .paths import ROOT, SKILLS_DIR, resolve


def summarize_archive(candidates: list, failures_shown: int = 4,
                      dominated_shown: int = 10) -> str:
    """Describe the workflows found so far, for the next round's prompt.

    The next round is asked to extend the frontier, not to tweak one incumbent,
    so it is handed the archive as raw material: each candidate's code and dev
    result, marked with whether it currently sits on the dev Pareto frontier, and
    — the part a scalar accuracy throws away — the dev examples it lost points on.
    Those per-example failures are what tell a new design where and why the
    existing ones break, which is the thing it has to beat.

    Every frontier candidate is always included; the dominated ones are capped at
    `dominated_shown` so the prompt stays bounded as the archive grows, keeping the
    most recent (the latest exploration, and usually the near-misses closest to the
    frontier). Whatever that drops is stated at the end rather than truncated in
    silence.

    Args:
        candidates: Every Candidate scored so far. Each needs `.dev` set. Empty on
            round 1, which returns "".
        failures_shown: How many of each candidate's worst dev examples to include.
        dominated_shown: How many off-frontier candidates to include besides the
            full frontier, most-recent first. 0 shows the frontier alone.

    Returns:
        The current frontier summarised first, then one block per shown candidate —
        name, a frontier mark, dev accuracy/cost, its code, and its worst dev
        examples — and a note of any dominated candidates omitted. Empty string
        when there are no candidates yet.
    """
    if not candidates:
        return ""
    # The candidates the agent sees are only ever dev-scored; test is held out.
    frontier = pareto_front(candidates, on=DEV)
    on_frontier = {id(c) for c in frontier}

    dominated = [c for c in candidates if id(c) not in on_frontier]
    kept_dominated = dominated[max(0, len(dominated) - dominated_shown):]  # most recent
    shown = on_frontier | {id(c) for c in kept_dominated}
    omitted = len(dominated) - len(kept_dominated)

    header = ["Current dev Pareto frontier (cheapest -> most accurate):"]
    header += [f"  {c.name}: accuracy {c.dev.accuracy:.2f}, ${c.dev.cost:.5f}/query"
               for c in frontier]

    blocks = []
    for c in candidates:                                     # shown ones, in proposal order
        if id(c) not in shown:
            continue
        mark = "  [ON FRONTIER]" if id(c) in on_frontier else ""
        parts = [f"### {c.name}{mark} — accuracy {c.dev.accuracy:.2f}, "
                 f"${c.dev.cost:.5f}/query, cached {c.dev.cached_input_frac:.0%}"]
        if c.description:
            parts.append(c.description)
        if c.dev.error:                        # never ran — show why, not empty code
            parts.append(f"(did not run: {c.dev.error})")
        parts.append("```python\n" + c.code.strip() + "\n```")
        digest = _failure_digest(c.dev, failures_shown)
        if digest:
            parts.append(digest)
        blocks.append("\n".join(parts))

    body = "\n".join(header) + "\n\n" + "\n\n".join(blocks)
    if omitted:
        body += (f"\n\n(+ {omitted} more dominated workflow(s) not shown — the "
                 f"{dominated_shown} most recent are kept; every frontier workflow is shown.)")
    return body


def _failure_digest(score, k: int) -> str:
    """The dev examples a candidate lost the most points on, formatted compactly.

    Its accuracy says how often it was wrong; this says on WHICH inputs and how —
    the answer it returned, the gold it was graded against, or the error that
    scored it 0. A numeric grader rejecting "150 miles", a router mis-sending hard
    inputs to the cheap model, a workflow blowing its call budget: all of that is
    here and none of it is in the scalar.

    Args:
        score: The candidate's dev SplitScore, whose `records` hold per-example
            question / gold / answer / score / error.
        k: How many of the worst examples to include.

    Returns:
        A short "Lost points on:" list, worst first, or "" when the candidate was
        perfect or never ran (nothing to show).
    """
    losers = sorted((r for r in score.records if r["score"] < 1.0),
                    key=lambda r: r["score"])[:k]
    if not losers:
        return ""
    lines = ["Lost points on:"]
    for r in losers:
        gold = _clip(str(r.get("gold", "")), 60)
        if r.get("error"):
            lines.append(f"  - Q: {_clip(r['question'], 160)} | gold: {gold} | "
                         f"ERROR: {_clip(r['error'], 120)}")
        else:
            lines.append(f"  - Q: {_clip(r['question'], 160)} | gold: {gold} | "
                         f"got: {_clip(str(r.get('answer', '')), 80)} | score {r['score']:.2f}")
    return "\n".join(lines)


def _clip(text: str, n: int) -> str:
    """One-line, length-capped view of a field for the archive summary.

    Args:
        text: The field to render.
        n: Longest result kept before an ellipsis is appended.

    Returns:
        `text` on one line, its runs of whitespace collapsed, truncated to `n`.
    """
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


AGENT_COST = re.compile(r"\[agent cost: \$([0-9.]+) over (\d+) turns\]")


def run_agent(agent_dir: pathlib.Path, log=print, on_cost=None) -> None:
    """Run the proposer subprocess in `agent_dir`, streaming its output to `log`.

    Shared by the research and design phases: both drive a Claude Agent SDK
    session the same way — spawn `flowopt.proposer` in the agent's
    scratch directory (its cwd), echo every line it prints, and report what the
    agent billed through the SDK. The caller stages the directory first and reads
    whatever the agent wrote afterward.

    Args:
        agent_dir: The scratch directory holding `proposer_config.json`.
        log: Where the agent's output goes, line by line.
        on_cost: Optional `on_cost(usd, turns)`, called with the agent's own spend.
            That goes through the SDK rather than our meter, so this is the only
            place it can be seen.
    """
    # The agent shells out to the eval skill, which imports flowopt;
    # both variables keep that working even if `python` on its PATH is not this
    # interpreter.
    src = str(ROOT / "src")
    env = {**os.environ, "PYTHONPATH": src, "FLOWOPT_SRC": src,
           # The agent session itself also runs through OpenRouter: Claude Code
           # speaks its native protocol to OpenRouter's Anthropic-compatible
           # endpoint, which is what lets ANY OpenRouter model be the designer.
           # ANTHROPIC_API_KEY must be explicitly blank so the auth token wins.
           "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
           "ANTHROPIC_AUTH_TOKEN": os.environ.get("OPENROUTER_API_KEY", ""),
           "ANTHROPIC_API_KEY": ""}
    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "flowopt.proposer"],
        cwd=agent_dir, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        line = line.rstrip()
        log(line)
        match = AGENT_COST.search(line)
        if match and on_cost:
            on_cost(float(match.group(1)), int(match.group(2)))
    process.wait()


def run_design_round(cfg, benchmark, round_num: int, context: str, log=print,
                     on_cost=None, research_notes: str = "",
                     run_skills_dir=None, guidance: str = "") -> list[dict]:
    """Run one design round and collect the workflows it proposed.

    Stages a scratch directory, runs the agent there as a subprocess, and streams
    its output to `log` as it goes.

    Args:
        cfg: The run config.
        benchmark: The Benchmark being optimized — supplies the task description,
            grading rule and dev examples the agent may test against.
        round_num: 1-based round number. Round 1 asks for a diverse initial set;
            later rounds hand over the whole archive and ask for new workflows
            that extend the Pareto frontier.
        context: `summarize_archive(...)` output. Ignored on round 1.
        log: Where the agent's output goes, line by line.
        on_cost: Optional `on_cost(usd, turns)`, called with what the agent spent
            on itself. That spend goes through the SDK rather than our meter, so
            this is the only place it can be seen.
        research_notes: The research phase's `research_notes.md`, or "". Passed to
            the round prompt so the designer builds on what was found for the task.
        run_skills_dir: When `designer.working_skills` is on, the run's persistent
            skills directory. Its skills are staged for the agent to read and any
            it writes this round are collected back into it for later rounds. None
            disables the feature.
        guidance: Operator guidance — free text from the human who reviewed the
            results so far, telling this round where to focus. "" adds nothing.

    Returns:
        The proposed programs as `{"name", "description", "code"}` dicts. Possibly
        empty if the agent produced nothing usable.
    """
    agent_dir = pathlib.Path(tempfile.mkdtemp(prefix=f"workflow_design_r{round_num}_"))
    _stage_agent_dir(cfg, benchmark, agent_dir, run_skills_dir)

    skills = list(cfg.designer.skills)
    if cfg.designer.working_skills and "workflow-skills" not in skills:
        skills.append("workflow-skills")     # teaches the agent to read/write working_skills/
    (agent_dir / "proposer_config.json").write_text(json.dumps({
        "model": cfg.designer.model,
        "skills": skills,
        "allowed_tools": list(cfg.designer.allowed_tools),
        "prompt": _round_prompt(cfg, benchmark, round_num, context, research_notes,
                                guidance),
    }))

    run_agent(agent_dir, log=log, on_cost=on_cost)
    if cfg.designer.working_skills and run_skills_dir is not None:
        built = _collect_skills(agent_dir, run_skills_dir)
        if built:
            log(f"  working skills after round {round_num}: {', '.join(built)}")
    programs = _collect_programs(agent_dir)
    helpers = _read_helpers(agent_dir)                 # the run's operators, if any
    for program in programs:
        program["helpers"] = helpers                   # snapshot what these workflows may call
    return programs


def _read_helpers(agent_dir: pathlib.Path) -> str:
    """Read the run's operator source (`working_skills/helpers.py`) the agent wrote.

    Functions defined here are injected before every candidate at eval time, so a
    workflow can call them by name. Empty when the agent wrote none (or the feature
    is off).

    Args:
        agent_dir: The directory the agent ran in.

    Returns:
        The contents of `working_skills/helpers.py`, or "".
    """
    path = agent_dir / "working_skills" / "helpers.py"
    return path.read_text() if path.exists() else ""


def _stage_agent_dir(cfg, benchmark, agent_dir: pathlib.Path, run_skills_dir=None) -> None:
    """Write everything the agent and its dev evaluator read from their cwd.

    Args:
        cfg: The run config, written out whole so the agent's evaluator meters
            candidates exactly as the search does — same models, same prices,
            same per-query limits.
        benchmark: Supplies the task description, grading rule and dev sample.
        agent_dir: The scratch directory to populate.
        run_skills_dir: When `designer.working_skills` is on, the run's persistent
            skills directory; its skills are copied into `working_skills/` for the
            agent to read (and extend). None or the flag off stages nothing extra.
    """
    (agent_dir / "run_config.yaml").write_text(OmegaConf.to_yaml(cfg))
    (agent_dir / "task_spec.json").write_text(json.dumps({
        "description": benchmark.description,
        "check": {"type": benchmark.grader.kind, "task": benchmark.description,
                  "rubric": benchmark.grader.rubric},
        # absolute: the agent reads this from its own scratch directory
        "grader": str(resolve(cfg.task.grader)) if cfg.task.grader else None,
    }))
    # The agent self-tests against TRAIN — the one slice it may see — so dev
    # stays unseen and its scores honest. A benchmark saved before the train
    # split existed (an old run being continued) falls back to a dev sample,
    # which is the old behavior, leak and all.
    train = benchmark.train or benchmark.dev[:cfg.designer.dev_sample_size]
    (agent_dir / "train_task.json").write_text(json.dumps(train))

    skills_dir = agent_dir / ".claude" / "skills"
    skills_dir.mkdir(parents=True)
    for name in cfg.designer.skills:
        shutil.copytree(SKILLS_DIR / name, skills_dir / name)

    if cfg.designer.working_skills:
        # The meta-skill teaches the agent to use working_skills/; the directory
        # itself carries whatever it wrote in earlier rounds of this run — both
        # SKILL.md notes and helpers.py operators — for it to read and extend. Kept
        # OUT of .claude/skills so an agent-written note with bad frontmatter can't
        # break the SDK's skill loading; the agent reads it via the meta-skill.
        if not (skills_dir / "workflow-skills").exists():    # may already be in designer.skills
            shutil.copytree(SKILLS_DIR / "workflow-skills", skills_dir / "workflow-skills")
        working = agent_dir / "working_skills"
        if run_skills_dir and pathlib.Path(run_skills_dir).exists():
            shutil.copytree(run_skills_dir, working)     # bring forward notes AND helpers.py
        else:
            working.mkdir()


def _collect_skills(agent_dir: pathlib.Path, run_skills_dir) -> list[str]:
    """Persist the skills the agent wrote this round into the run's skills directory.

    The agent reads and writes `working_skills/` in its scratch directory, and that
    scratch is discarded, so anything it wrote is copied back here to survive into
    the next round of the SAME run (the directory is fresh per run).

    Args:
        agent_dir: The directory the agent ran in.
        run_skills_dir: The run's persistent skills directory to copy into.

    Returns:
        What that directory now holds — one entry per SKILL.md note (its folder
        name), plus "helpers.py" when the agent wrote operators — so the round
        can say what the agent has built up rather than leaving it to be
        discovered from the filesystem. Empty when the agent wrote nothing.
    """
    written = agent_dir / "working_skills"
    if not written.exists():
        return []
    run_skills_dir = pathlib.Path(run_skills_dir)
    run_skills_dir.mkdir(parents=True, exist_ok=True)
    # Mirror everything back — SKILL.md notes and helpers.py alike — so the next
    # round reads and extends what this one wrote.
    shutil.copytree(written, run_skills_dir, dirs_exist_ok=True)
    built = sorted(p.parent.name for p in run_skills_dir.glob("*/SKILL.md"))
    if (run_skills_dir / "helpers.py").exists():
        built.append("helpers.py")
    return built


def _round_prompt(cfg, benchmark, round_num: int, context: str,
                  research_notes: str = "", guidance: str = "") -> str:
    """Build the prompt for one design round.

    Args:
        cfg: The run config, for the model list.
        benchmark: The task being optimized.
        round_num: 1-based round number; round 1 gets the "design a diverse set"
            goal, later rounds the "extend the frontier" one. A continued run's
            round 1 carries the source archive in `context`, so it gets the
            improve goal despite the number.
        context: `summarize_archive(...)` output, used whenever non-empty.
        research_notes: The research phase's findings, inlined so the designer
            builds on them. "" when the research phase was skipped or found nothing.
        guidance: Operator guidance — the human who reviewed the results so far
            says where to focus. Inlined near the end so it reads as the freshest
            instruction. "" adds nothing.

    Returns:
        The full prompt for the agent.
    """
    from .models import ModelCatalog                 # here to avoid an import cycle
    analysis = benchmark.analysis
    # What the agent routes between: each pool model with its OpenRouter price
    # and Artificial Analysis measurements, so delegation decisions rest on
    # measured capability, not on what the agent assumes about a name.
    catalog = ModelCatalog.from_config(cfg)
    menu = "\n".join(f"- {catalog.describe(model_id)}" for model_id in catalog.ids)
    # What the agent needs to return an answer in the right SHAPE, not just with
    # the right value — the grader is strict, so "42 apples" scores 0 on a numeric
    # task. The examples come from the analyzer.
    facts = ("Available models (all served via OpenRouter), cheapest -> most "
             "expensive, with Artificial Analysis capability indices where "
             "measured:\n" + menu + "\n\n"
             f"solve() must RETURN its final answer, scored by check="
             f"'{benchmark.grader.kind}'. Nothing is parsed out of prose.")
    if benchmark.grader.kind == "numeric":
        facts += " A numeric answer must BE the number, with no words or units around it."
    elif benchmark.grader.kind == "exact":
        facts += " It is compared against the gold answer exactly, ignoring case."
    elif benchmark.grader.kind == "llm_equality":
        facts += (" It is scored by a binary equality checker: 1 only if the final "
                  "answer matches the reference (small tolerance on numbers), 0 "
                  "otherwise — no partial credit, so return the exact answer alone.")
    if analysis.answer_examples:
        facts += ("\nCorrectly formatted answers for this task look exactly like:\n- "
                  + "\n- ".join(str(e) for e in analysis.answer_examples[:4]))
    # Tell the agent what tools it may design with. A workflow that calls a
    # forbidden one is rejected, so a wrong assumption here wastes candidates.
    tools = list(cfg.runtime.tools or [])
    if tools:
        described = {"web_search": "web_search searches the web",
                     "web_fetch": "web_fetch retrieves a URL already in the prompt",
                     "subagent": ("subagent lets the model delegate a self-contained "
                                  "subtask to a worker model mid-call")}
        facts += (f"\n\nServer-side tools workflows MAY use: {', '.join(tools)}. "
                  "Pass them via tools=[...] on call_model. "
                  + "; ".join(described[t] for t in tools if t in described) + ". "
                  "There is NO code_execution tool — models cannot run code, so exact "
                  "arithmetic must come from the workflow's own Python, not from a tool.")
    else:
        facts += ("\n\nNo server-side tools are available for this task: workflows must "
                  "NOT pass tools= to call_model. It is closed-book — no web access.")
    # The budget only ever filtered the final recommendation, which meant the
    # agent could spend the whole search designing workflows nobody would pick.
    facts += (f"\n\nCost target: the workflow that gets recommended must cost no more "
              f"than ${cfg.report.max_cost_per_query:.5f} per query. Designing above it "
              f"is not wasted — an expensive workflow that is much more accurate is "
              f"still worth knowing about — but at least one candidate should come in "
              f"under it.")

    if research_notes.strip():
        facts += ("\n\nResearch notes on what works for THIS task (gathered by web "
                  "research before designing) — build on these, and don't spend "
                  "candidates rediscovering what they report as wasted effort:\n"
                  + research_notes.strip())

    if cfg.designer.working_skills:
        facts += ("\n\nYou have a working_skills/ directory (see the workflow-skills "
                  "skill): read any notes there and build on them; record reusable "
                  "lessons as skills; and write reusable operators in "
                  "working_skills/helpers.py that your solve() code can then call by "
                  "name — they are injected into the workflow, like extract_last_number.")

    if guidance.strip():
        facts += ("\n\nOPERATOR GUIDANCE — the human running this search reviewed the "
                  "results so far and asks the next designs to focus on:\n"
                  + guidance.strip())

    # An archive means improve-the-frontier, whatever the round number says — a
    # continued run starts at round 1 but carries the source run's archive.
    goal = (prompts.render("design_goal_initial", description=benchmark.description)
            if round_num == 1 and not context.strip() else
            prompts.render("design_goal_improve", description=benchmark.description,
                           archive=context))
    return prompts.render("design_round", goal=goal, facts=facts)


def _collect_programs(agent_dir: pathlib.Path) -> list[dict]:
    """Read the agent's chosen workflows out of its scratch directory.

    Args:
        agent_dir: The directory the agent ran in.

    Returns:
        `{"name", "description", "code"}` dicts from `programs.json`. If the agent
        never wrote that file — it ran out of turns, or errored — any candidate
        `.py` files it left behind are salvaged instead, rather than throwing the
        whole round away.
    """
    programs_file = agent_dir / "programs.json"
    if programs_file.exists():
        data = json.loads(programs_file.read_text())
        return data["programs"] if isinstance(data, dict) else data

    salvaged = []
    for path in sorted(agent_dir.glob("*.py")):
        code = path.read_text()
        if "def solve" in code:
            salvaged.append({"name": path.stem, "description": "(recovered)", "code": code})
    return salvaged
