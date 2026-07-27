"""Run one design-agent session. Entry point for the subprocess `designer` spawns.

Run from the agent's scratch directory (its cwd): it reads `proposer_config.json`
from there and drives a Claude Agent SDK session. The agent — via the skills
staged under `./.claude/skills/` — writes its picks to `programs.json`.

    python -m flowopt.proposer
"""
import asyncio
import json
import os
import sys

from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ResultMessage,
                              TextBlock, ToolUseBlock, query)

# The input field worth echoing for tools not special-cased below, tried in this
# order: a search is its query, a skill its name. Whatever hits first is the one
# thing a reader wants on the line.
_TOOL_DETAIL_KEYS = ("command", "file_path", "path", "query", "url", "pattern",
                     "skill", "description", "prompt")


def _clip(text: str, n: int) -> str:
    """Collapse text to one whitespace-normalized line of at most `n` chars."""
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + "…"


def _tool_line(block) -> str:
    """Render one tool call as a log line that says what the tool actually did.

    `[tool] Bash: python eval_candidate.py c1.py` reads; `[tool] Bash` doesn't.
    The file tools get their own shapes — a Write is its path plus how much was
    written, an Edit its path plus the size of the change — and paths are shown
    relative to the agent's scratch directory (this process's cwd), because the
    twelve identical temp-dir prefixes were burying the one part that differs.
    Anything unrecognized falls back to its raw input as JSON, so a tool line is
    never bare when the call had arguments at all.

    Args:
        block: A ToolUseBlock — its `input` dict holds the call's arguments.

    Returns:
        The line, detail collapsed to one line and clipped.
    """
    inp = getattr(block, "input", None) or {}
    name = block.name
    cwd = os.getcwd() + os.sep

    def rel(path) -> str:
        path = str(path or "")
        return path[len(cwd):] if path.startswith(cwd) else path

    if name == "Bash":
        detail = _clip(inp.get("command", ""), 300)
    elif name == "Write":
        detail = f"{rel(inp.get('file_path'))} ({len(inp.get('content') or '')} chars)"
    elif name == "Edit":
        detail = (f"{rel(inp.get('file_path'))} "
                  f"(-{len(inp.get('old_string') or '')} +{len(inp.get('new_string') or '')} chars)")
    elif name == "Read":
        detail = rel(inp.get("file_path"))
    else:
        detail = ""
        for key in _TOOL_DETAIL_KEYS:
            value = inp.get(key)
            if isinstance(value, str) and value.strip():
                detail = _clip(value, 240)
                break
        if not detail and inp:      # arguments of some other shape — show them raw
            detail = _clip(json.dumps(inp, default=str), 200)
    return f"  [tool] {name}" + (f": {detail}" if detail else "")


def _turn_preview(content) -> str:
    """The first ~100 chars of whatever a turn contains, for its marker line.

    Scans blocks in order and previews the first one with substance — reply
    text, thinking (which is otherwise not logged at all), or the first tool
    call — so the markers alone read as a story of the session.

    Args:
        content: An AssistantMessage's content blocks.

    Returns:
        A one-line clipped preview, or "" for an empty turn.
    """
    for block in content:
        kind = type(block).__name__
        if kind == "TextBlock" and (block.text or "").strip():
            return _clip(block.text, 100)
        if kind == "ThinkingBlock" and (getattr(block, "thinking", "") or "").strip():
            return "thinking: " + _clip(block.thinking, 90)
        if kind == "ToolUseBlock":
            return _clip(_tool_line(block).strip(), 100)
    return ""


async def main() -> None:
    """Drive one agent session to completion, echoing its progress to stdout.

    Reads `proposer_config.json` from the current directory: the model to run,
    the skills to load, the tools to allow, and the prompt. The agent's own
    output is the side effect — files it writes into the working directory,
    chiefly `programs.json`.

    Raises:
        Exception: Anything the SDK raises. The caller below turns it into a
            non-zero exit so the search can salvage what the agent left behind.
    """
    cfg = json.loads(open("proposer_config.json").read())
    cap = cfg.get("max_turns")
    turn = 0
    options = ClaudeAgentOptions(
        model=cfg["model"],
        cwd=os.getcwd(),
        setting_sources=["project"],            # discovers ./.claude/skills/<name>/
        skills=cfg["skills"],
        allowed_tools=cfg["allowed_tools"],
        permission_mode="bypassPermissions",
        # The session bills per turn on a growing context; uncapped means an
        # uncapped bill. The caller sizes this to the phase.
        max_turns=cfg.get("max_turns"),
    )
    async for message in query(prompt=cfg["prompt"], options=options):
        if isinstance(message, AssistantMessage):
            # Every agent turn lands in the log, numbered against the cap and
            # carrying a content preview, so the markers alone show the pace,
            # the remaining budget, and what each turn was about.
            turn += 1
            preview = _turn_preview(message.content)
            print(f"— turn {turn}" + (f"/{cap}" if cap else "")
                  + (f" · {preview}" if preview else "") + " —", flush=True)
            for block in message.content:
                if isinstance(block, TextBlock):
                    # The run log shows this as "the agent's own output", so keep
                    # enough to read its reasoning — and say when it was cut,
                    # rather than ending mid-sentence as if that were all.
                    text = block.text
                    if len(text) > 2000:
                        text = text[:2000] + f" [… clipped {len(block.text) - 2000} chars]"
                    print(text, flush=True)
                elif isinstance(block, ToolUseBlock):
                    print(_tool_line(block), flush=True)
        elif isinstance(message, ResultMessage):
            # The design agent bills through the SDK, not through our meter, so
            # this is the only place its spend is observable. Printed in a fixed
            # form because the caller reads it back off the log.
            cost = message.total_cost_usd
            print(f"[agent finished: {message.subtype}]", flush=True)
            if cost is not None:
                print(f"[agent cost: ${cost:.4f} over {message.num_turns} turns]", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:                  # let the caller salvage candidates
        print(f"[proposer error] {type(error).__name__}: {error}", flush=True)
        sys.exit(1)
