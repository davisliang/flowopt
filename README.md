# flowopt

Automatically find the best **LLM workflow** for a task under a budget. Instead of
hand-writing prompting strategies, a design agent **writes workflow programs** and
**iterates on them to cut cost without losing accuracy**, runs them on your data, and
hands you the accuracy/cost **Pareto frontier** to pick from.

```sh
export OPENROUTER_API_KEY=...            # every model call goes through OpenRouter
export ARTIFICIAL_ANALYSIS_API_KEY=...   # optional: capability indices in the picker & prompts
uv sync
uv run flowopt-ui --open                        # the UI, at :8770
uv run flowopt --task gsm8k                     # or headless
uv run flowopt --task gsm8k designer.rounds=1   # override any config key
```

In Python, `Session` is the entry point — it holds the config, the model catalog
derived from it, and the client:

```python
from flowopt import Session, analysis, optimize, report

session = Session.load("gsm8k", ["designer.rounds=1"])
benchmark = analysis.build_benchmark(session.cfg, session.client)
search = optimize(session.cfg, benchmark, session.evaluator(benchmark.grader))
report.summarize(search, session.cfg)
```

## The UI

`uv run flowopt-ui` serves a page on `127.0.0.1:8770` that starts
searches, watches them run, and compares what they found:

- **Cost estimate before you commit** — the form estimates what a search will
  cost, broken down and with every assumption listed. **Measure this task** runs
  a few real calls (a few cents, seconds) and estimates from that instead: it
  pins this task's token scale and, by grading its own answers, whether the cheap
  model can do the work at all — which is what decides whether the designer
  escalates. Calibrated against two measured searches; ARC predicts within 4% of
  what it actually cost, ifeval within 2x.
- **New search** — either pick a **benchmark** (the 14 routerllm holdout tasks
  plus ARC-AGI-2, with their example counts, graders and recorded baselines), or
  **describe a task** in free text and optionally upload your own `.jsonl`.
  Upload nothing and the examples are generated. **Train / dev / test counts** set
  each split's size explicitly (seeded random sampling); blank splits the task's
  pool by the configured fraction. **Models** — pick the pool the workflows may
  route over from everything on OpenRouter (343 models and counting), each shown
  with its OpenRouter price and Artificial Analysis intelligence / coding / math
  indices and speed, and pick the **design agent model** the same way; the agent
  is told the pool's measurements and researches how those models compare for
  the task before designing. **Tools workflows may use** — OpenRouter's server
  tools web_search, web_fetch and subagent (in-call delegation to a worker
  model, off by default; there is no server-side code execution) — are
  checkboxes; uncheck them all for a closed-book task so no candidate can look
  answers up, and its numbers stay comparable to a closed-book baseline.
  **Request plugins** — response-healing (auto-repair malformed JSON) and
  context-compression (middle-out truncation), applied to every call, both off
  by default because they trade correctness visibility for convenience.
  **Skills** show as
  checkboxes too: the core set is locked on (the design prompt drives the agent
  through those by name, so dropping one would buy a round of malformed
  candidates), any skill you add under `skills/<name>/SKILL.md` appears as an
  optional extra, and a **working skills** toggle lets the agent keep a
  run-scoped skills folder (notes and `helpers.py` operators) it extends across
  rounds. Only the listed settings are accepted from the form; everything else
  comes from config.
- **Live progress** — phase pills (analyzing → researching → designing round *i* →
  ranking → done), the research notes the agent wrote for the task, candidates
  appearing with dev accuracy and cost as they are scored, and the raw log including
  the design agent's own output. **Stop** kills the run and the agent with it.
- **Results** — an accuracy-vs-cost plot with the frontier drawn through it, a
  candidate table with dev and test scores side by side, and each workflow's
  source on click.
- **Continue a search** — a finished run takes more rounds, with an optional
  **nudge**: free text telling the next designs where to focus. The archive,
  research notes, working skills and the exact dev/test splits carry over
  (the benchmark is saved with the run, so generated data is never re-generated),
  and carried test scores aren't paid for twice.
- **Reading a run** — a convergence chart (best dev accuracy after each round;
  a flat segment says that round bought nothing, which is the continue-or-stop
  signal), a spend bar splitting the money between the research agent, the
  design agent, and dev/test scoring, and round toggles on the accuracy/cost
  chart so a continuation's carried archive and each round's additions can be
  read separately.
- **Every input and output** — pick a candidate, then its **diagram** — the
  workflow drawn from its traces, one row per distinct sequence of model calls
  with its frequency, mean score and cost, so a router's branch rates and an
  escalator's escalation rate read directly off the rows — or **dev calls** /
  **test calls**: per example, its score, what the workflow returned, the gold
  answer, and each model call in order with the full prompt sent, the reply
  received, the model, the cost and the token split. This is how you tell "the
  strategy is wrong" from "the model got that one wrong". Prompts and replies
  over 8k chars are clipped, with the full length shown.
- **Same example, every workflow** — a matrix lining up every candidate's answer
  to the same question side by side, most-disagreed-on examples first. Accuracy
  says which workflow won; this says where they differed and whether the
  expensive one is right for a reason.
- **The rest of the run** — the grading rule and judge rubric, the answer format
  shown to the designer, sample dev and test examples, the resolved config, a
  timeline of milestones, and the full log.
- **Compare** — every candidate from every run on one accuracy-vs-cost chart
  (log cost axis), coloured by task, hover any point for its name, task, scores
  and description, click to jump to it. Filter to one task and routerllm's
  haiku / opus / router / oracle accuracies are drawn as reference lines.

Runs live in `runs/<run_id>/` — the resolved config, a status header, an
append-only event log, the raw log, the research notes, the saved benchmark
(the exact splits, so a continuation scores the same data), per-candidate call
traces, the run's working skills, and the result. Clicking the run's name in
the UI opens this folder in the file browser.
The server holds no state of its own: it reads those files, and each search runs
in its own subprocess. Restart the server mid-search and the page picks up where
it was.

It binds to localhost because starting a search spends real money — anything that
can reach the port can spend it.

## Benchmarks

`benchmarks/<name>/` is a self-contained task: `benchmark.yaml` (what it is, how
it grades, routerllm's baselines) plus `data.jsonl` of `{"question", "answer"}`.
`config/task/<name>.yaml` is generated alongside, so each is usable as
`--task <name>` or from the UI.

The 14 routerllm holdout tasks and ARC-AGI-2 are imported by:

```sh
uv run python scripts/import_routerllm_benchmarks.py          # --limit 200 by default
```

Large sets are sampled deterministically (`random.Random(0).sample`) and the
yaml records `sampled_from`, so the sampling is never silent. Baselines are
**recomputed** from routerllm's `joined_14.jsonl` rather than copied, which is
checkable: the import reproduces ifeval's known 0.848 / 0.891 / 0.848 / 0.957.

The import labels every row with routerllm's own **80/10/10 train/val/test
partition** — `split.json` (the router-dataset wandb artifact, pulled into the
routerllm checkout) keys by prompt hash, and the paired dataset joins that hash
to the doc ids the exports use. The optimizer then draws each of its splits
ONLY from the matching partition: train from their train, dev from their val,
and test from their test — the holdout the recorded baselines were measured on
— so no example ever crosses a partition boundary and a test score is computed
on the same examples as the baselines. Coverage is stated per benchmark
(`split_labeled`, `holdout_in_data`); it is partial where the export was itself
a sample of the full task.

Grading is mapped onto ours where it can be: `exact` → exact match, `contains` →
`benchmarks/_graders/contains.py`, `grid` → `benchmarks/_graders/grid.py` (both
ported from routerllm so a score means the same thing), `judge` → our LLM judge.

Other tasks are built the same way by their own scripts, each freezing a
`benchmark.yaml` + `data.jsonl` + `config/task/<name>.yaml` and its grader:
`scripts/make_game24.py` (a solver-generated puzzle set), `scripts/import_fanoutqa.py`
(fan-out multi-hop QA, plus a judge-graded variant), `scripts/build_news_current.py`
/ `scripts/build_news_multihop.py` (recent-news QA frozen from a sourced,
timestamped `sources.json`), and `scripts/build_ml_papers.py` (problem→solution
pairs extracted from last month's arXiv ML abstracts, keeping only the pairs
whose problem and approach are both clearly stated; judge-graded against each
paper's actual approach). The time-sensitive benchmarks carry their freeze date
in their **name** — `news_current_20260721`, `ml_papers_<as_of>` — so the name
itself says when the data goes stale and should be recycled; refreshing the
sources and re-running a builder writes a new stamped benchmark beside the old
one.

Three tasks are graded natively by machinery that needs more than a prompt and
an answer. `ifeval` works through the existing checker in `experiments/`.
`humaneval_plus_gen` and `mbpp_plus` need a sandboxed test harness we don't have,
so they are imported as a small reference sample with
`grading_supported: false` and are not offered as runnable tasks. **A `judge`
task's numbers are not comparable to the recorded baselines** — routerllm judges
with its own prompts, we use ours; `grading_note` says so per benchmark.

## Vocabulary

These words mean one thing each, everywhere — in the code, the config, and below.

| Term | Meaning |
| --- | --- |
| **workflow** | A Python program `solve(question, call_model) -> answer`. The thing being optimized. |
| **candidate** | One workflow the design agent proposed, plus how it scored. |
| **benchmark** | A task made measurable: its analysis, a grader, and dev/test splits. |
| **task analysis** | What the task is and how an answer should be graded, inferred from your description. |
| **grader** | Scores one answer in [0, 1] — numeric, exact, LLM-judge, or a task's own metric. |
| **split score** | One candidate's accuracy and cost on one split. |
| **search** | One optimization: every candidate tried, and the finalists. |
| **train / dev / test** | Train is the only slice the design agent may see (self-tests, few-shot material). Dev guides the search. Test is held out; only the final ranking touches it. |
| **frontier** | The non-dominated candidates — nothing else is both cheaper and more accurate. |
| **call meter** | The per-query object whose `call_model` a workflow calls. Measures and caps. |

## The idea in one paragraph

A workflow is an arbitrary Python function `solve(question, call_model) -> answer`.
Because it's *code*, it can express **any** inference-time paradigm — a single call,
chain-of-thought, self-consistency, decomposition, debate, a cheap→expensive router —
without the harness needing a special case for each. The harness fixes only three
things, and all the generality rides on them:

1. **Contract** — every workflow is `solve(question, call_model) -> answer`. It
   *returns* its answer, so nothing has to be parsed back out of prose.
2. **Metered call site** — `call_model(prompt, model=...)` is the *only* way a
   workflow can reach a model. It counts tokens, prices them, and enforces a
   per-query budget, so cost is measured at one chokepoint no matter what the code
   does. It returns a `Reply` — a string carrying the full response, so a run can be
   inspected afterwards without anything being discarded.
3. **Task-inferred grading** — a `Grader` (numeric tolerance / exact match /
   LLM-judge / the task's own metric), so the evaluator never needs to know the
   paradigm.

## The pipeline

1. **Analyze the task** (`analysis`) — one structured call infers a task description,
   the grading rule, and — rather than picking from a fixed menu — a **judge rubric**
   for free-form tasks, which is then calibrated against example answers and dropped
   for a generic judge if it doesn't discriminate. `dataset` generates labeled
   examples if you didn't supply any, split into **dev** and held-out **test**. A task
   that already knows its own shape sets `task.description` in config and skips this.
2. **Research the task** (`research`) — before any workflow is written, a **Claude
   Agent SDK** agent searches the web for how this kind of task is best approached,
   reads as many sources as it needs, and writes `research_notes.md`. Those notes are
   handed to every design round, so designs build on known-good techniques rather than
   only on what the model already carries in its weights. Skipped with
   `designer.research=false`.
3. **Design and optimize** (`designer`, `optimizer`) — a **Claude Agent SDK** agent
   runs once per round, driven by the skills below and the research notes. Round 1
   designs a diverse initial set; each later round is shown the **archive so far** —
   every frontier workflow's code, its dev accuracy and cost, and the dev examples it
   got wrong, plus recent dominated near-misses (`designer.dominated_shown` caps how
   many) — and asked for new ones that **extend the accuracy/cost frontier**: cheaper at
   a given accuracy, more accurate at a given cost, or filling a gap (a cheaper model,
   fewer calls, difficulty routing, exact computation in solve()'s own Python). Feeding
   back the per-example failures, not just a scalar accuracy, is what tells a new design
   where the current ones break. Every candidate is scored on dev and added to the archive.
   With `designer.working_skills=true` the agent also keeps a **run-scoped `working_skills/`
   folder** it reads and writes across rounds: `SKILL.md` notes on what it learned, and
   reusable operators in `helpers.py` that its workflows call by name (injected into the
   sandbox like `extract_last_number`).
4. **Rank finalists** — the dev frontier is re-scored on the **held-out test split**,
   for numbers nothing was tuned against.
5. **Choose** (`report`) — the frontier, the two constrained picks (*the best workflow
   I can afford*, *the cheapest one that's good enough*), a plot, and each finalist's
   code.

## Layout

```
config/                 every knob (OmegaConf)
  config.yaml           models (OpenRouter ids), call, runtime, judge, data, designer, report
  task/*.yaml           one file per task: the seed prompt, optional data + grader
prompts/*.md            every prompt sent to a model, as text (${placeholders})
skills/                 what the design agent is taught
  workflow-research/    research what works for a task, write research_notes.md
  workflow-design/      the methodology and the program contract
  workflow-eval/        the dev evaluator (a wrapper over the same runtime)
  workflow-naming/      naming workflows by structure, so results tables compare
  workflow-skills/      (opt-in) read/write run-scoped skill notes + callable operators
src/flowopt/
  config.py             typed config schema + loading/overrides
  session.py            Session: config + catalog + client, wired once
  catalog.py            the model feeds: OpenRouter (prices) + Artificial Analysis
                        (capability / speed indices), disk-cached and slug-joined
  models.py             ModelCatalog: ids, prices, capabilities — built from the feeds
  client.py             ModelClient — the one place anything reaches a model
                        (OpenRouter chat completions; the design agent runs through
                        OpenRouter's Anthropic-compatible endpoint)
  prompts.py            fills prompts/*.md
  grading.py            Grader: numeric / exact / llm_judge / custom
  runtime.py            Reply, CallMeter, compile_solve, Evaluator, SplitScore
  analysis.py           what is this task, and how is an answer graded
  dataset.py            load the task's examples, or generate diverse ones
  research.py           research the task online, write research_notes.md
  designer.py           stage and run one design round (and the shared agent runner)
  proposer.py           the agent subprocess entry point
  optimizer.py          Candidate, Search — the round loop and the archive
  pareto.py             frontier + the two constrained picks
  report.py             frontier table, plot, search JSON
  runstore.py           one run's state on disk: status, events, log, result
  cli.py                `flowopt`
  dashboard/            the UI: stdlib server, runner subprocess, one static page
runs/<run_id>/          per-run state the UI reads (gitignored)
experiments/            benchmark comparisons (see routerllm_ifeval/)
tests/                  everything checkable without spending money
```

## Configuration

Everything tunable lives in `config/`, nothing in the code. A task is one small file:

```yaml
# config/task/gsm8k.yaml
task:
  name: gsm8k
  seed_prompt: >-
    Grade-school math word problems. Each answer is a single integer.
data:
  n_examples: 120
```

Override any key from the command line — `uv run flowopt --task gsm8k
designer.rounds=1 runtime.concurrency=4 report.max_cost_per_query=0.001` — or in
Python via `Session.load("gsm8k", [...])`. The schema in
`src/flowopt/config.py` is typed, so a misspelled key fails at load time
instead of being silently ignored.

Optional task fields: `dataset` (a `.jsonl` of `{"question", "answer"}`), `grader`
(a `.py` exposing `grade(prediction, item) -> float`, to score with an external
benchmark's own metric), `description` (skip the analyzer), and — for
`check_type: llm_judge` — `judge_rubric` (a known-shape task's own grading
criteria, still calibrated before use, instead of letting the analyzer infer one).

## The metered runtime

Generated programs are model-written code, so each one runs through
`flowopt.runtime`:

- **Metered** — every `call_model()` call adds to a per-query token and cost tally.
- **Capped** — `runtime.max_model_calls` and `runtime.max_tokens` per query; a program
  that blows its budget or crashes scores 0 on that example rather than sinking the run.
- **Sandboxed** — candidates run with a restricted import list and builtins allowlist
  (see `compile_solve`). The allowlist is deliberately generous: a name it refuses
  doesn't read as "blocked" in the results, it reads as "this strategy scores 0". This
  raises the bar; it is **not** a security boundary — run genuinely untrusted code in a
  container. When run-scoped operators are enabled, the agent's `helpers.py` is executed
  into the same namespace before each candidate, under the same restrictions, so a
  workflow can call its operators by name.

The design agent's dev evaluator (`skills/workflow-eval/eval_candidate.py`) is a thin
**wrapper over this same code**, handed the same resolved config, so a dev number means
what it will mean in the final ranking.

## Notes

- Grading returns a **score in [0, 1]**; a candidate's accuracy is the mean over the
  dataset. Nothing is parsed out of prose: `numeric` requires the answer to *be* a
  number and the judge replies under a schema. `numeric`/`exact` are 1/0; `llm_judge`
  is a **graded** score from a cheap model against a task-specific rubric, so for
  free-form tasks "accuracy" is mean quality. The judge's API calls are the
  *evaluator's* cost, deliberately **not** counted as workflow cost.
- Costs are **cache-aware**: every call sets a prompt-cache breakpoint, so the same
  prompt resent to the same model bills cache reads (~90% off the input rate), while a
  different model never shares the cache. Caching only engages above a per-model size
  floor (~1–4k tokens), so `SplitScore.cached_input_frac` reports the share of input
  tokens that actually came from cache — a workflow built on "resending is cheap" can
  otherwise pay full price with nothing to say so.
- The final ranking is always on the held-out test split, so candidates aren't scored
  on data the design agent tuned against.

## Tests

```sh
uv run pytest
```

The API is faked, so the suite costs nothing. `test_offline.py` covers the pipeline's
pure logic — pricing, grading, the answer contract, the Pareto helpers, the runtime's
guardrails (sandbox, call budget, crash isolation), config overrides, and what the
design agent is handed. `test_dashboard.py` covers the UI's run store and what the
server refuses to accept from the form.
