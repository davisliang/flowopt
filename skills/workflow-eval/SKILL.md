---
name: workflow-eval
description: Score a candidate solve(question, call_model) workflow program on the train set. Use to test any candidate before keeping it — it reports accuracy and cost per query, or the error if the program is broken.
---

# Evaluate a workflow candidate

Test a candidate workflow program (a `.py` file defining `solve(question, call_model)`;
see the **workflow-design** skill for the contract) against the train set — the examples you are allowed to see; the dev split the search scores stays unseen.

## How to run

Run from the working directory — it must contain `run_config.yaml`,
`task_spec.json` and `train_task.json` (the run's settings, the task's grading spec,
and the train examples). Invoke the bundled script on the candidate file:

```bash
python .claude/skills/workflow-eval/eval_candidate.py <candidate_file.py>
```

The script rebuilds the task's checker from `task_spec.json` (numeric /
exact-match / LLM-judge / the task's own grader) and runs the candidate on every
dev example through the same metered, sandboxed runtime the final search uses —
it is a wrapper over that code, not a copy of it, so a dev number means what it
will mean in the final ranking.

## Output

A single JSON line:

- Success: `{"ok": true, "accuracy": <0..1>, "cost_per_query": <usd>, "n": <count>, "cached_input_frac": <0..1>, "errors": [...], "failures": [...]}`
- Failure: `{"ok": false, "error": "<message>"}`

Read it as: `accuracy` is the mean score on the train set, `cost_per_query` is the
measured mean USD cost, and `cached_input_frac` is the share of input tokens that
came from the prompt cache — if you designed for cache reuse and this is 0, the
prompts are below the cache floor and the reuse isn't happening. `ok: false`
means the program didn't compile or didn't define `solve`. A non-empty `errors`
array with `ok: true` means some examples raised — investigate and fix the
candidate. Keep candidates that run cleanly; fix or drop the rest.

`failures` is the diagnosis channel: the train examples the candidate lost points
on, worst first (up to 25), each with the question, the gold answer, what the
candidate answered, its score, and the error if it raised instead. The search's
dev results come back to you as aggregates only, so THIS is where you learn
whether a design fails on formatting, on a difficulty band, on a topic, or on
its own routing — read it before designing the fix, not after.
