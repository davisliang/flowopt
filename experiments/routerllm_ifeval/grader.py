"""ifeval grading, using lm-eval's own checker.

Validated: reproduces the recorded correct{haiku,opus} labels on all 146
ifeval examples (292/292 checks), so a score here means the same thing it
means in routerllm's summary.json.

Kept OUT of the workflow sandbox on purpose. A program that could call this
would be checking its answer against the grader itself, which is benchmark
gaming — the constraints are stated in the prompt, so a workflow has to
verify them on its own.
"""
import sys

LMEVAL_SITE_PACKAGES = (
    "/Users/davis/Documents/code/routerllm/.lmeval/lib/python3.12/site-packages"
)
if LMEVAL_SITE_PACKAGES not in sys.path:
    sys.path.append(LMEVAL_SITE_PACKAGES)

import langdetect  # noqa: E402

# ifeval's checkers call langdetect for the language, all-lowercase and
# all-uppercase constraints, and langdetect randomises its detector per process
# unless seeded. Neither lm-eval nor routerllm pins it, so grading the SAME
# answers twice can give different scores. Pinned here so a result is
# reproducible; see README for the size of the effect.
langdetect.DetectorFactory.seed = 0

from lm_eval.tasks.ifeval.utils import (  # noqa: E402
    InputExample,
    test_instruction_following_strict,
)

def grade(prediction, item):
    # prompt_level_strict_acc: 1.0 only if EVERY constraint on the prompt holds.
    doc = item["doc"]
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    try:
        out = test_instruction_following_strict(inp, str(prediction))
    except Exception:
        return 0.0        # a malformed answer fails the constraints, it doesn't crash the run
    return float(out.follow_all_instructions)

def per_constraint(prediction, item):
    # Diagnostic only: which individual constraints held. Never used for scoring.
    doc = item["doc"]
    inp = InputExample(key=doc["key"], instruction_id_list=doc["instruction_id_list"],
                       prompt=doc["prompt"], kwargs=doc["kwargs"])
    try:
        out = test_instruction_following_strict(inp, str(prediction))
    except Exception:
        return {i: False for i in doc["instruction_id_list"]}
    return dict(zip(doc["instruction_id_list"], out.follow_instruction_list, strict=True))
