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

Steps in execution order, separated by `→`. Each step is a short model name plus
modifiers. Lowercase single words are non-model operations.

```
models      the model's SHORT NAME — the distinctive tail of its id, unique
            within the pool. For the default pool:
              v4-flash   deepseek/deepseek-v4-flash
              v4-pro     deepseek/deepseek-v4-pro
              kimi       moonshotai/kimi-k3
              glm        z-ai/glm-5.2
              muse       meta/muse-spark-1.1
              luna       openai/gpt-5.6-luna        luna-pro    its pro serving
              terra      openai/gpt-5.6-terra       terra-pro   its pro serving
              sol        openai/gpt-5.6-sol         sol-pro     its pro serving
              opus       anthropic/claude-opus-5
              sonnet     anthropic/claude-sonnet-5
              gem-3.1-pro    google/gemini-3.1-pro-preview
              gem-3.5-flash  google/gemini-3.5-flash
              gem-3.6-flash  google/gemini-3.6-flash
              gem-3.5-lite   google/gemini-3.5-flash-lite
            A model outside this table gets the shortest id-tail that is unique
            in the run's pool. Never a bare letter — with 17-model pools the
            letters collide and the names stop being readable — and never the
            full provider path, which is noise.
modifiers   ^  high effort           ~  medium effort
            #  web search            ×N N samples of that step
ops         vote  pick  first  stop  skip        (lowercase = no model call)
branch      {decider: A|B}   choose between A and B
sugar       ?X  ==  {self: skip|X}   run X only if the previous step is unsure
```

### Branches name their decider

`{decider: A|B}` — whatever decides which arm runs goes before the colon:

```
{re: sonnet#|v4-flash}       a regex over the INPUT picks     (cannot see the answer)
{self: stop|sonnet^}         the previous step's own verdict  (only as good as that step)
{glm: opus^|v4-flash}        a GLM classifier picks
```

This is the field most worth making explicit. A branch is only as good as its
decider, and two routers with the same shape but different deciders behave nothing
alike. Writing `{...}` without naming the decider hides the variable that usually
explains the result.

## Full identifier: `task/notation@vN`

The notation alone is not unique. It deliberately omits the prompt, so two
programs with the same shape and different prompts collide — and
`v4-flash→sonnet` designed for `ifeval` is a different program from the same
shape designed for `mmlu_pro`. Qualify it:

```
ifeval/v4-flash→sonnet@v1               first flash-draft/sonnet-audit built for ifeval
ifeval/v4-flash→sonnet@v2               same structure, different audit prompt
mmlu_pro/sonnet×5→vote@v1               unrelated to anything above
hle/v4-flash→v4-flash→{self: stop|sol^}@v1
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
v4-flash                             one deepseek-v4-flash call
v4-flash→sonnet                      flash drafts, sonnet audits and repairs
v4-flash→v4-flash                    flash drafts, flash audits itself
sonnet→opus^                         sonnet drafts, opus audits at high effort
v4-flash→sonnet#                     flash drafts, sonnet audits with web search
sonnet×5→vote                        five sonnet samples, majority vote
glm×3→vote→?sol                      vote, escalate to sol when the vote is split
v4-flash→{re: sonnet#|v4-flash}      regex routes retrieval prompts to sonnet+web
v4-flash→v4-flash→{self: stop|sol^}  flash self-audits; escalate to sol if unsure
luna→{glm: terra|stop}               a cheap glm check decides if a terra audit runs
```

## Rules

- **Execution order, always.** The name reads the way the program runs.
- **One short name per model, from the table above.** Every model in a name must
  be identifiable without a per-run legend; two workflows from different runs
  should be comparable by name alone.
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
last step of each name gave the entire result: pipelines ending in `sonnet` or
`opus` scored 0.957, those ending in the cheap drafting model scored at or below
the no-audit baseline, and everything containing `#` was worse still. No
commentary needed.
