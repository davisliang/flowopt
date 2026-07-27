"""Step 1.5 — research what works for this task, before any workflow is written.

A mandatory web-research phase. A Claude Agent SDK session, driven by the
`workflow-research` skill, studies the task online — reading as many sources as
it needs — and records what it found in `research_notes.md`. Those notes are
then handed to every design round, so the designer builds on known-good
approaches for the task rather than only on what the model already carries in its
weights. The notes are saved with the run so a reader can see them too.

Like the designer, the agent runs in a SEPARATE PROCESS
(`flowopt.proposer`) — see `designer.py` for why a subprocess.
"""
import json
import pathlib
import shutil
import tempfile

from . import prompts
from .analysis import Benchmark
from .designer import run_agent
from .paths import SKILLS_DIR

# The skill staged into the research agent's `.claude/skills/`.
RESEARCH_SKILL = "workflow-research"


def run_research(cfg, benchmark: Benchmark, log=print, on_cost=None) -> str:
    """Run the research phase and return the notes it produced.

    Stages a scratch directory with the research skill, runs the agent there with
    web access, and reads back `research_notes.md`.

    Args:
        cfg: The run config. Uses `cfg.designer.model` to run the agent and
            `cfg.designer.allowed_tools` for what it may reach — the same list the
            designer gets, which includes WebSearch and WebFetch.
        benchmark: The task being researched — supplies its description.
        log: Where the agent's output goes, line by line.
        on_cost: Optional `on_cost(usd, turns)`, called with the agent's own spend.

    Returns:
        The contents of `research_notes.md`, or "" if the agent wrote none (it ran
        out of turns, errored, or genuinely found nothing to record).
    """
    agent_dir = pathlib.Path(tempfile.mkdtemp(prefix="workflow_research_"))

    skills_dir = agent_dir / ".claude" / "skills"
    skills_dir.mkdir(parents=True)
    shutil.copytree(SKILLS_DIR / RESEARCH_SKILL, skills_dir / RESEARCH_SKILL)

    from .models import ModelCatalog                 # here to avoid an import cycle
    catalog = ModelCatalog.from_config(cfg)
    menu = "\n".join(f"- {catalog.describe(model_id)}" for model_id in catalog.ids)
    (agent_dir / "proposer_config.json").write_text(json.dumps({
        # Research reads and summarizes; the cheap model does that fine, and the
        # designer only ever sees the notes.
        "model": cfg.designer.research_model or cfg.designer.model,
        "skills": [RESEARCH_SKILL],
        "allowed_tools": list(cfg.designer.allowed_tools),
        "max_turns": int(cfg.designer.research_max_turns),
        "prompt": prompts.render("research_task", description=benchmark.description,
                                 models=menu,
                                 max_turns=int(cfg.designer.research_max_turns)),
    }))

    run_agent(agent_dir, log=log, on_cost=on_cost)
    return _collect_notes(agent_dir)


def _collect_notes(agent_dir: pathlib.Path) -> str:
    """Read `research_notes.md` out of the agent's scratch directory.

    Args:
        agent_dir: The directory the agent ran in.

    Returns:
        The notes, or "" if the agent never wrote the file.
    """
    notes = agent_dir / "research_notes.md"
    return notes.read_text() if notes.exists() else ""
