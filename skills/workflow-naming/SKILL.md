---
name: workflow-naming
description: Name an inference-time workflow after what it DOES, not what it was meant to achieve. Use when naming candidate programs, comparing workflows, or reporting results, so two programs that look alike in a results table actually are alike.
---

# Naming workflows by structure

A workflow's name should let a reader reconstruct the program from the name, and
let two names be compared without opening either file.

Names that describe *intent* fail at this. In one real run, `crossmodel_opus_audit`,
`router_haiku_selfcheck_escalate` and `router_haiku_codeexec_by_constraint` sounded
like a family and shared almost nothing; meanwhile `haiku_draft_sonnet_audit` and
`haiku_draft_sonnet_higheffort_audit` sounded like two strategies and differed by a
single argument on one line. The results table was unreadable as a result — the
finding "only the auditor's model matters" was invisible until the names were
rewritten.

## The notation

Steps in execution order, separated by `→`. Each step is a model letter plus
modifiers. Lowercase words are non-model operations.

```
models      one capital letter per pool model, assigned cheapest -> most
            expensive; with the default pool: H = claude-haiku-4.5,
            L = gpt-5.6-luna, T = gpt-5.6-terra, S = claude-sonnet-5,
            O = claude-opus-4.8, G = gpt-5.6-sol
modifiers   ^  high effort           ~  medium effort
            #  web search            ×N N samples of that step
ops         vote  pick  first  stop  skip        (lowercase = no model call)
branch      {decider: A|B}   choose between A and B
sugar       ?X  ==  {self: skip|X}   run X only if the previous step is unsure
```

### Branches name their decider

`{decider: A|B}` — whatever decides which arm runs goes before the colon:

```
{re: S#|H}        a regex over the INPUT picks           (cannot see the answer)
{self: stop|S^}   the previous step's own verdict picks  (only as good as that step)
{S: O^|H}         a Sonnet classifier picks
```

This is the field most worth making explicit. A branch is only as good as its
decider, and two routers with the same shape but different deciders behave nothing
alike. Writing `{...}` without naming the decider hides the variable that usually
explains the result.

## Full identifier: `task/notation@vN`

The notation alone is not unique. It deliberately omits the prompt, so two
programs with the same shape and different prompts collide — and `H→S` designed
for `ifeval` is a different program from `H→S` designed for `mmlu_pro`. Qualify it:

```
ifeval/H→S@v1                     first H→S built for ifeval
ifeval/H→S@v2                     same structure, different audit prompt
mmlu_pro/S×5→vote@v1              unrelated to anything above
ifeval/H→H→{self: stop|S^}@v1
```

- **`task/`** — the benchmark task the program was designed for. Programs are
  tuned per task and are not comparable across tasks even when identically shaped.
- **`@vN`** — bumps when the structure is unchanged but the internals differ:
  prompt wording, schema fields, a threshold, a regex. This is the escape hatch for
  everything the notation drops on purpose. Assign in first-seen order per task.
- `/` and `@` are reserved for this and appear nowhere else in the notation, so
  the identifier splits unambiguously. (`:` is already taken by `{decider: ...}`.)

Record the code's SHA alongside the identifier in results. The name is for reading;
the hash is what makes a row reproducible. Two rows sharing a name and differing in
hash means someone forgot to bump the version.

Identifiers are not filenames — `→` and `|` are fine in a `name` field and not in a
path. Keep candidate `.py` files named however you like.

## Examples

```
H                     one Haiku call
H→S                   Haiku drafts, Sonnet audits and repairs
H→H                   Haiku drafts, Haiku audits itself
S→O^                  Sonnet drafts, Opus audits at high effort
H→S^                  Haiku drafts, Sonnet audits at high effort
H→H#                  Haiku drafts, Haiku audits using code execution
S×5→vote              five Sonnet samples, majority vote
H×3→vote→?O           vote, escalate to Opus when the vote is split
H→{re: S#|H}          regex sends counting prompts to Sonnet+code, rest to Haiku
H→H→{self: stop|S^}   Haiku self-audits; escalate to Sonnet only if unsure
H→{S: S|stop}         a cheap Sonnet check decides whether a Sonnet audit runs
```

## Rules

- **Execution order, always.** The name reads the way the program runs.
- **One character per model.** Add letters to the legend if the model set grows;
  do not spell model names out.
- **Length tracks complexity.** `H` is one character because it is one call. If a
  name is long, the program is complicated, and that should be visible.
- **Do not encode intent.** No `smart`, `robust`, `cheap`, `best`, `crossmodel`,
  `hardpath`. If the name would still fit after the program changed, it is wrong.
- **Do not encode the prompt.** Two `H` programs with different system prompts are
  both `H`. Prompt differences are real and can be large — track them in a separate
  column, not by lengthening the name.
- Report the name next to accuracy and cost, so a results table can be scanned
  structurally: every pipeline ending in `S`, every one containing `#`, and so on.

## Why this pays off

Sorting a results table by these names groups programs that behave alike, because
the name is a description of behaviour. In the run this came from, reading only the
last step of each name gave the entire result: pipelines ending in `S` or `O` scored
0.957, those ending in `H` scored at or below the no-audit baseline, and everything
containing `#` was worse still. No commentary needed.
