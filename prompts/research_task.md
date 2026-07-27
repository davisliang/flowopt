Research what works for this task before any workflow is designed:
${description}

Use the workflow-research skill. Search the web (this is required), read as many
sources as you need within your ${max_turns}-turn budget, and write your
findings to research_notes.md in the working directory. A larger model — the
design agent — will read ONLY your notes, never your sources, so write them as
a clear, dense briefing it can act on directly.

The workflows will route between these models (all served via OpenRouter), with
prices and Artificial Analysis capability indices where measured:
${models}

Structure the briefing in two parts:

1. THE TASK — how this kind of task is best approached: known-good strategies,
   failure modes, formatting pitfalls, and anything specific to this benchmark.

2. THE MODELS — a short, clear summary of EACH model above: what it is strong
   and weak at for this kind of task, practical experience from independent
   evaluations and community reports, reliability at instruction-following and
   formatting, and whether it punches above or below its price here. The design
   agent decides which model does which step from your summaries, so make each
   one concrete — not a restatement of the menu numbers.

Keep enough of your turn budget to actually write the notes — a briefing that
never gets written is worth nothing.

Do not design or test any workflows — your only output is research_notes.md.
Write it and stop.
