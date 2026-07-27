---
name: workflow-design
description: Design and select a diverse set of inference-time LLM workflows for a task, as runnable Python programs that span the cost/accuracy tradeoff. Use when asked to propose, design, or optimize LLM workflows/strategies for a dataset.
---

# Design inference-time LLM workflows

Produce a diverse set of candidate workflows for the task described in the
prompt, each as a Python program, and keep only the ones that actually work.
Span the cost/accuracy tradeoff — from a cheap single call to elaborate
multi-call paradigms (chain-of-thought, self-consistency / majority vote,
decompose-then-solve, debate, difficulty routing, cheap-model-with-escalation).

## The program contract

Each candidate is a `.py` file defining exactly:

```python
def solve(question, call_model):
    ...
    return answer
```

- **`solve` must RETURN the final answer itself** — nothing is parsed out of prose.
  Return the bare value (the number, the label, the text), not a sentence wrapped
  around it: return `"42"`, not `"The answer is 42."`. The prompt states the
  `check` rule that scores it.
    - You may instead return a dict `{"answer": <the answer>, ...}` if you want to
      keep extra context alongside it. Only `answer` is graded — a returned object
      with no `answer` key is a contract violation and scores 0 with an error.
- `call_model(prompt, model=None, system=None, tools=None, effort=None, schema=None)`
  is the ONLY way to call a model. It returns a `Reply`, which **is** a string (so
  `return call_model(p)` works), with the full response attached: `.blocks` (every
  content block, including tool calls and their results), `.data` (parsed JSON when
  you passed a `schema`), `.usage`, `.model`.
    - `model=<name>` — route to a specific model (see `MODELS`, cheap → expensive).
    - `system="..."` — set a system prompt for that call.
    - `tools=[...]` — server-side tools; they run on the API side and the results
      come back in the same reply. `"web_search"` searches the web, `"web_fetch"`
      reads a URL already present in the prompt, `"subagent"` lets the model
      delegate a self-contained subtask to a worker model mid-call (prefer routing
      `call_model` yourself — it keeps each step's cost visible and tunable). The
      prompt says which are allowed for THIS task — calling any other fails the
      query. There is NO code-execution tool: models cannot run code, so exact
      arithmetic or string manipulation belongs in your own solve() Python, which
      runs for free.
    - `effort="low"|"medium"|"high"|"xhigh"|"max"` — sets how deeply the model
      thinks, on models that support reasoning (the prompt's model list says
      which; ignored elsewhere). On a short prompt whose output is
      schema-constrained, the model often skips thinking entirely, and then
      `effort="high"` behaves exactly like no effort at all — same answer, same
      cost. Check `reply.usage["output"]`: if a high-effort call's output tokens
      match a plain call's, no thinking happened. To make reasoning actually
      engage, say so in the prompt ("this requires multi-step reasoning — think
      carefully before answering"), or reason in a schema-free call first and
      format the answer in a second one. Thinking bills as output tokens; the
      per-query budget still applies.
    - `schema=<JSON Schema>` — constrain the reply to JSON matching it, and read the
      parsed object off `reply.data`. This is the most reliable way to get a clean
      answer out of a model, and it composes with `tools=` (the schema shapes only
      the text the model writes at the end). If the model refuses, `reply.data`
      is `None` — fall back to the reply text.
      Use the provided **`ANSWER_SCHEMA`** for the call that produces the final
      answer; it is `{"answer": <string>}`, the shape `solve` is graded on. Give
      intermediate calls whatever schema fits them — a difficulty router wants
      `{"difficulty": ...}`, a decomposer `{"subquestions": [...]}`. Only the value
      you RETURN has to carry an `answer`.
- Inside `solve` you may use, with no imports: `re`, `json`, `statistics`,
  `Counter`, `extract_last_number(text) -> float | None`, the list `MODELS`, and
  `ANSWER_SCHEMA`.
- **Candidates run sandboxed.** No file / network / system access inside `solve`,
  and `import` is limited to this list — importing anything else fails the
  candidate at compile time, so stay inside it:
  `re`, `json`, `math`, `statistics`, `collections`, `itertools`, `functools`,
  `string`, `random`, `time`, `typing`, `dataclasses`, `textwrap`, `operator`,
  `heapq`, `difflib`, `decimal`, `fractions`.
  Ordinary Python is otherwise fine — classes, comprehensions, `getattr`, and
  `try/except` over the usual exception types all work. There is no `open`,
  `eval`, `exec`, `os`, `sys`, or `subprocess`.
- There is no output-length knob: every call gets one generous ceiling. You pay for
  the tokens a reply actually uses, so length is controlled by the prompt, not a cap.
- The runtime meters cost and enforces a per-query call/token budget, so keep the
  number of model calls modest.

A reliable shape for the last step of a workflow:

```python
def solve(question, call_model):
    reply = call_model(question, schema=ANSWER_SCHEMA)
    return reply.data["answer"] if reply.data else str(reply).strip()
```

Returning the reply itself also works — `return call_model(question, schema=ANSWER_SCHEMA)`
is unwrapped to its `answer` for you.

## Improving existing workflows

If the prompt gives you existing workflows with their accuracy and cost, your job
is to make them **cheaper without losing accuracy**. Good moves: use a cheaper
model, make fewer model calls, route easy inputs to the cheap model and only
escalate hard ones, or move exact computation (arithmetic, parsing, counting)
into solve()'s own Python so the model never has to be sampled for it. Keep a
new candidate only if it stays at least as accurate as the best existing
workflow while costing less per query.

**Prompt caching only engages above a per-model size floor — check before relying
on it.** Resending the same prompt to the same model bills the repeat at ~10% of the
input rate, but ONLY if the shared prefix is long enough. Below the floor nothing is
cached and there is no error, just a silently full-price call:

| model family | shared prefix must exceed |
|---|---|
| `anthropic/*` haiku and opus tiers | ~4,096 tokens |
| `anthropic/*` sonnet tiers | ~1,024 tokens |
| `openai/*` | ~1,024 tokens |
| other providers | varies; assume ~1,024+ and verify |

(Everything is served via OpenRouter, which passes provider caching through —
OpenAI-style models cache automatically, Anthropic-style ones use the cache
breakpoints the runtime already sets. The floor logic applies either way.)

Short-prompt tasks are the common case and they are all far below this — a one-line
question with a paragraph of system prompt is ~100 tokens, so caching cannot help at
all. Do NOT choose a workflow shape on the theory that repeating a prompt is cheap
unless the prefix is genuinely long (a big system prompt, few-shot examples, a
document). Verify rather than assume: `reply.usage["cache_read"]` is 0 when nothing
cached, and the runtime reports the cached share of input tokens for the whole run.

Output tokens are never cached, so even when caching does engage it cuts
self-consistency's *input* cost but not its output cost, and a *different* model
never shares the cache.

## Workflow

1. Optionally research inference-time techniques with WebSearch / WebFetch.
2. Write each candidate to its own `.py` file in the working directory.
3. Test candidates **one at a time, in the foreground**, with the **workflow-eval**
   skill. Do NOT launch background jobs (`&`), job control, or many evals at once —
   idle-waiting on background jobs makes the session hang and drop the connection.
   Run one eval, read its JSON result, then move to the next. Fix or drop
   candidates that error.
4. Keep **4–5** diverse, WORKING candidates spanning cheap → accurate. One dev run
   per candidate is enough — don't re-test.
5. **Name each candidate by its structure**, using the **workflow-naming** skill —
   `v4-flash→sonnet`, `sonnet×5→vote`, `v4-flash→{self: stop|sol^}`. Names describe
   what the program does, not what you hoped it would achieve, so the results
   table can be compared by name.
6. **Your final action MUST be to write `programs.json`** in the working directory:
   a JSON list of objects with keys `name`, `description`, `code` (code = the full
   `solve` source). Include every candidate that passed; exclude any that errored.
   Do this even if a candidate was slow — never end the session without writing
   `programs.json`.
