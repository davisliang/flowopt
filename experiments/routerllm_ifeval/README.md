# routerllm ifeval — workflow optimization vs. a trained router

Optimizes a workflow for the `ifeval` task of routerllm's 14-dataset holdout
benchmark, and reports it against that repo's recorded baselines.

## Why this task

`ifeval` scores `prompt_level_strict_acc`: an answer counts only if it satisfies
EVERY formatting constraint stated in the prompt. On routerllm's numbers the
trained router scores 0.848 there — identical to `haiku_only`, i.e. it never
escalates — while the routing oracle reaches 0.957. Opus is *worse* than Haiku
on the train slice (0.810 vs 0.840), so capability isn't the lever; compliance
is. And because the constraints are stated in the prompt, a workflow can verify
its own answer before returning it, which is not something routing can express.

## Comparability

Grading uses lm-eval's own `test_instruction_following_strict`, the same checker
behind routerllm's `summary.json`. `build_data.py` verifies it against that
repo's recorded `correct{haiku,opus}` labels and refuses to write data if the
reported slice disagrees:

    test  92/92   agree   <- reported; asserted exact
    train 197/200 agree   <- selection only

The train gap is the benchmark disagreeing with itself, not a bug here. ifeval's
`change_case:*` and `language:*` checkers call `langdetect`, which randomises per
process unless seeded, and neither lm-eval nor routerllm seeds it — so grading
identical answers twice can give different scores. `grader.py` pins
`DetectorFactory.seed = 0`.

Measured effect, regrading the recorded answers under 12 seeds:

    train  n=100   spread 0.010-0.020   (1-2 examples)
    test   n= 46   spread 0.0000        (stable)

So the reported numbers carry no grading noise; only selection is affected. The
46-example test slice is a different matter for *sampling* noise — one example
is worth 2.2 points, so differences under ~4 points are not meaningful.

`grader.py` is deliberately kept OUT of the workflow sandbox. A program able to
call the checker would be grading itself against the metric.

## Data

    selection : 100 train examples  (split.json train partition, seed 0)
    reporting : 46 test examples    (holdout14 <=100/task subset)

Zero overlap between them; the June-23 split's test partition is an exact set
match to `holdout14_examples.jsonl`.

## Run

Build the two slices first — they are derived from the routerllm checkouts and are
not committed here, so nothing runs until this has been done once:

    uv run python build_data.py --out .     # writes ifeval_{train,test}.jsonl

Then:

    OPENROUTER_API_KEY=... uv run python -u run_ifeval.py --rounds 3
    uv run python -u run_ifeval.py --smoke # 8 examples, 1 round, proves the wiring

Use `-u`: the design agent's subprocess output is block-buffered otherwise and
the run looks hung when it isn't.

The task itself — description, data, grader, and a tighter per-query budget than
the default — is `config/task/ifeval.yaml`; this script only adds the reporting
against routerllm's recorded baselines. Everything else comes from `flowopt`.

Paths to the routerllm checkout are absolute in `grader.py` — adjust if the
sibling repo moves.
