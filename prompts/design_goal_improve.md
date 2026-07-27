Extend the accuracy/cost Pareto frontier for this task:
${description}

Below is EVERY workflow tried so far: its code, and its accuracy and cost on the
search's dev split (aggregates only — dev examples themselves are never shown).
The ones on the current frontier are marked [ON FRONTIER] — those are the points
a new design has to beat.

${archive}

Design 3-4 NEW workflows that would land ON the frontier — each one either
cheaper than the current frontier at the same accuracy, more accurate at the
same cost, or filling a gap between two frontier points. To see WHERE and WHY a
design loses points, run it through the workflow-eval skill: it scores against
the train examples and reports the per-example failures (question, gold, what it
answered). Diagnose there, then fix the failure mode — and borrow and combine
ideas from any workflow above, not just the most accurate one.
