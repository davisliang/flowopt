"""Step 3 — research the task, loop the design agent, then rank on held-out test.

When `designer.research` is set, a web-research phase runs first and its
`research_notes.md` is handed to every round. Each round the agent is shown every
workflow tried so far — code, dev scores, and
the examples each got wrong — and asked for new ones that extend the frontier;
every candidate is scored on dev and added to the archive. Only the dev Pareto
frontier is re-scored on test — those are the candidates actually worth choosing
between, and test calls cost money.
"""
import pathlib
import tempfile
from dataclasses import dataclass, field
from typing import Optional

from . import designer, research
from .analysis import Benchmark
# DEV / TEST are re-exported here for callers' convenience: this is the module
# that defines Candidate, the thing they select a split of.
from .pareto import DEV, TEST, pareto_front
from .runtime import Evaluator, SplitScore
from .session import Session


@dataclass
class Candidate:
    """One workflow program the design agent proposed, and how it scored.

    Attributes:
        name: Structural name, e.g. "H×3→vote→?S^". Unique within a search.
        description: The agent's one-line summary of what the workflow does.
        code: The program's source — a `solve(question, call_model)` definition.
        helpers: The run's shared operator source (`working_skills/helpers.py`)
            this workflow may call, snapshotted when it was designed. Injected
            before `code` at eval time. "" when no operators were written.
        dev: Its SplitScore on the dev split, set once scored.
        test: Its SplitScore on held-out test. Only finalists have one.
    """
    name: str
    description: str
    code: str
    helpers: str = ""
    dev: Optional[SplitScore] = None
    test: Optional[SplitScore] = None

    @property
    def program(self) -> dict:
        """The `{"name", "code", "helpers"}` dict `Evaluator.run` takes."""
        return {"name": self.name, "code": self.code, "helpers": self.helpers}


@dataclass
class Search:
    """The result of one optimization: everything tried, and the finalists.

    Attributes:
        archive: Every distinct candidate seen, in the order proposed. All are
            dev-scored.
        finalists: The dev Pareto frontier, re-scored on held-out test. These are
            the workflows worth choosing between.
    """
    archive: list[Candidate] = field(default_factory=list)
    finalists: list[Candidate] = field(default_factory=list)

    def test_frontier(self) -> list[Candidate]:
        """The finalists still non-dominated on their held-out test scores.

        The finalists are the DEV frontier re-scored on test, and test can
        reshuffle them — a candidate that led on dev may be dominated on test.
        Everything reported as "the frontier" recomputes it on the test numbers
        through this one method.
        """
        return pareto_front(self.finalists, on=TEST)


def rebuild_search(result: dict, trace_for=None) -> Search:
    """Rebuild a Search's archive from a saved result, to continue a run.

    The archive is what later rounds design against, so a continued search
    starts from everything the source run already paid to learn — including,
    when traces are available, the per-example dev failures that tell a new
    design where the old ones break.

    Args:
        result: A parsed `result.json` (report.save's output).
        trace_for: Optional `trace_for(name)` returning the candidate's saved
            dev trace dict (`runstore.read_trace` shape), or None. Its records
            feed the archive summary's failure digest; without it the carried
            candidates still count, they just can't show WHERE they lost points.

    Returns:
        A Search whose archive holds every saved candidate that has code, all
        dev-scored; test scores are carried where the source ranked them.
        Finalists are left empty — the continued run re-derives its frontier.
    """
    search = Search()
    for saved in result.get("candidates", []):
        if not saved.get("code"):
            continue          # recovered-from-events entries can't be re-run or deduped
        candidate = Candidate(name=saved["name"], description=saved.get("description", ""),
                              code=saved["code"], helpers=saved.get("helpers", ""))
        dev = saved.get("dev") or {}
        trace = trace_for(saved["name"]) if trace_for else None
        records = [{"question": r["question"]["text"], "gold": r["gold"]["text"],
                    "answer": r["answer"]["text"], "score": r["score"],
                    "cost": r.get("cost", 0.0), "error": r.get("error"), "calls": []}
                   for r in (trace or {}).get("records", [])]
        candidate.dev = SplitScore(name=candidate.name,
                                   accuracy=dev.get("accuracy", 0.0),
                                   cost=dev.get("cost_per_query", 0.0),
                                   cached_input_frac=dev.get("cached_input_frac", 0.0),
                                   records=records)
        test = saved.get("test")
        if test:
            candidate.test = SplitScore(name=candidate.name, accuracy=test["accuracy"],
                                        cost=test["cost_per_query"])
        search.archive.append(candidate)
    return search


def optimize(cfg, benchmark: Benchmark, evaluator: Evaluator = None, log=print,
             on_event=None, on_scored=None, on_research=None, skills_dir=None,
             search: "Search" = None, research_notes: str = "",
             guidance: str = "") -> Search:
    """Research the task, design candidates, score them on dev, rank on test.

    Args:
        cfg: The run config; `cfg.designer.rounds` sets how many rounds run.
        benchmark: What is being optimized — task, grader, and the two splits.
        evaluator: Scores candidates. Built from `cfg` and the benchmark's grader
            if omitted.
        log: Where progress goes, as human-readable lines.
        on_event: Optional `on_event(dict)` called at each milestone —
            "round_start", "candidate", "ranking", "test_scored". Where `log` is
            prose for a terminal, these are structured for a UI to render.
        on_scored: Optional `on_scored(candidate, split, score)` called as soon as
            a candidate is scored, with the SplitScore itself. That carries the
            per-example records and every model call, which are too large for the
            event stream — a caller that wants them writes them somewhere.
        on_research: Optional `on_research(notes)` called once with the research
            phase's `research_notes.md` text, so a caller can persist it. The notes
            can be large, which is why they go through a callback rather than the
            event stream.
        skills_dir: Where to keep the run's `working_skills/` when
            `designer.working_skills` is on — the design agent reads and extends it
            across rounds. Defaults to a fresh temp dir (discarded with the run); a
            caller can point it at the run directory to keep the skills for
            inspection. Ignored when the feature is off.
        search: An existing Search to fill in. Pass one and the caller keeps a
            reference to the archive as it grows, so a run that is stopped or
            crashes half way can still report what it had already scored — a
            long search represents real money and losing it is not acceptable.
            A search whose archive is already populated (a continued run) is
            treated as prior rounds: round 1 sees it in the archive summary.
        research_notes: Already-written research notes to hand every round.
            Non-empty skips the research phase — a continued run reuses the
            source run's notes instead of paying for the phase again.
        guidance: Operator guidance for the design rounds — free text from the
            human who reviewed earlier results, telling the next designs where
            to focus. "" adds nothing.

    Returns:
        A Search. `finalists` is empty when no candidate survived.
    """
    evaluator = evaluator or Session.from_config(cfg).evaluator(benchmark.grader)
    emit = on_event or (lambda event: None)
    scored = on_scored or (lambda candidate, split, score: None)
    save_research = on_research or (lambda notes: None)
    search = search if search is not None else Search()

    run_skills_dir = None
    if cfg.designer.working_skills:
        run_skills_dir = pathlib.Path(skills_dir) if skills_dir else pathlib.Path(
            tempfile.mkdtemp(prefix="workflow_run_skills_"))
        run_skills_dir.mkdir(parents=True, exist_ok=True)

    if research_notes:
        log(f"reusing research notes ({len(research_notes)} chars)")
    elif cfg.designer.research:
        log("\n===== research =====")
        emit({"event": "researching"})
        research_notes = research.run_research(
            cfg, benchmark, log=log,
            on_cost=lambda usd, turns: emit(
                {"event": "agent_cost", "round": 0, "usd": usd, "turns": turns}))
        save_research(research_notes)
        log(f"research notes: {len(research_notes)} chars")
        emit({"event": "researched", "chars": len(research_notes)})

    for round_num in range(1, cfg.designer.rounds + 1):
        log(f"\n===== design round {round_num} / {cfg.designer.rounds} =====")
        emit({"event": "round_start", "round": round_num, "rounds": cfg.designer.rounds})
        context = designer.summarize_archive(          # empty on round 1
            search.archive, cfg.designer.failures_shown, cfg.designer.dominated_shown)
        report_cost = lambda usd, turns: emit(
            {"event": "agent_cost", "round": round_num, "usd": usd, "turns": turns})
        for program in designer.run_design_round(cfg, benchmark, round_num, context,
                                                 log=log, on_cost=report_cost,
                                                 research_notes=research_notes,
                                                 run_skills_dir=run_skills_dir,
                                                 guidance=guidance):
            # Skip exact repeats. A candidate's behavior is its code AND the
            # operators it may call, so the same code resubmitted after
            # helpers.py changed is a new candidate, not a repeat.
            if any(program["code"] == c.code and program.get("helpers", "") == c.helpers
                   for c in search.archive):
                continue
            candidate = Candidate(name=_unique_name(program["name"], search.archive),
                                  description=program.get("description", ""),
                                  code=program["code"],
                                  helpers=program.get("helpers", ""))
            candidate.dev = evaluator.run(candidate.program, benchmark.dev)
            scored(candidate, "dev", candidate.dev)
            search.archive.append(candidate)
            log(f"  + {candidate.name:24s} dev acc {candidate.dev.accuracy:.2f}  "
                f"${candidate.dev.cost:.5f}/query  "
                f"cached {candidate.dev.cached_input_frac:.0%}")
            emit({"event": "candidate", "round": round_num, "name": candidate.name,
                  "description": candidate.description,
                  "dev_accuracy": candidate.dev.accuracy, "dev_cost": candidate.dev.cost,
                  "cached_input_frac": candidate.dev.cached_input_frac,
                  "errors": candidate.dev.errors[:3]})

    log(f"\n{len(search.archive)} workflows in the archive after {cfg.designer.rounds} rounds.")
    if not search.archive:
        return search

    log("scoring the dev frontier on the held-out test split...")
    frontier = pareto_front(search.archive, on=DEV)
    emit({"event": "ranking", "n_finalists": len(frontier)})
    for candidate in frontier:
        fresh = candidate.test is None
        if fresh:
            candidate.test = evaluator.run(candidate.program, benchmark.test)
            scored(candidate, "test", candidate.test)
        # else: carried from the source run — the same held-out split was
        # already paid for, and re-scoring would only add sampling noise.
        search.finalists.append(candidate)
        log(f"  {candidate.name:24s} test acc {candidate.test.accuracy:.2f}  "
            f"${candidate.test.cost:.5f}/query" + ("" if fresh else "  (kept)"))
        emit({"event": "test_scored", "name": candidate.name,
              "test_accuracy": candidate.test.accuracy, "test_cost": candidate.test.cost,
              # kept scores were paid for by the source run; the spend view skips them
              "kept": not fresh})
    return search


def _unique_name(name: str, archive: list[Candidate]) -> str:
    """Make a proposed name unique within the archive.

    Two rounds can propose the same structural name for different code; keeping
    them distinguishable stops a results table from merging them.

    Args:
        name: The name the agent gave the program.
        archive: Candidates already accepted.

    Returns:
        `name`, or `name#2`, `name#3`, ... if it is already taken.
    """
    taken = {c.name for c in archive}
    if name not in taken:
        return name
    suffix = 2
    while f"{name}#{suffix}" in taken:
        suffix += 1
    return f"{name}#{suffix}"
