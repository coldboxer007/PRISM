"""Dimension 3: Retrospective Metacognitive Report prompt templates.

D3a (blind self-assessment): Before seeing results, the model predicts
    which steps it got right/wrong based on introspection alone.
D3b (informed reflection): After seeing results, the model identifies
    the hardest step and provides counterfactual analysis.
"""


def build_blind_retrospective_prompt(num_steps: int) -> str:
    """Build the blind self-assessment prompt (D3a).

    Delivered after the model has solved the problem but BEFORE
    revealing which steps were correct/incorrect.  The model must
    introspect on its own reasoning to judge each step.

    Args:
        num_steps: number of steps in the problem
    """
    step_template = "\n".join(f"   Step {i + 1}: [label]" for i in range(num_steps))

    return f"""Before I tell you how you did, I want you to reflect on your own \
reasoning.

PER-STEP SELF-ASSESSMENT: For each of the {num_steps} steps you just worked \
through, categorize your experience as exactly one of:
   - "confident and correct"   (you feel sure you got it right)
   - "confident but wrong"     (you feel sure but suspect an error)
   - "uncertain and correct"   (you're unsure but think it might be right)
   - "uncertain and wrong"     (you're unsure and suspect an error)

Format as:
{step_template}"""


def build_informed_retrospective_prompt(
    step_results: list[bool],
    overall_correct: bool,
) -> str:
    """Build the informed retrospective prompt (D3b).

    Delivered after the blind self-assessment.  Reveals which steps were
    correct/incorrect, then asks for hardest step identification and
    counterfactual analysis (used for T4 coherence scoring).

    Args:
        step_results: list of booleans, True if step was correct
        overall_correct: whether the final answer was correct
    """
    n = len(step_results)
    correct_count = sum(step_results)

    step_lines = "\n".join(
        f"Step {i + 1}: {'CORRECT' if r else 'INCORRECT'}"
        for i, r in enumerate(step_results)
    )

    overall_str = "CORRECT" if overall_correct else "INCORRECT"

    return f"""Here is your actual result: You got {correct_count} out of {n} steps correct.
Specifically:
{step_lines}
Your final answer was {overall_str}.

Now reflect on your performance:

1. HARDEST STEP: Which step actually gave you the most trouble \
during your reasoning? Give just the number.

2. COUNTERFACTUAL: Name one specific thing you considered doing \
differently but chose not to. What was the alternative approach, \
and why did you reject it?"""
