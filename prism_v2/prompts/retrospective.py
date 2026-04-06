"""Dimension 3: Retrospective Metacognitive Report prompt template."""


def build_retrospective_prompt(
    step_results: list[bool],
    overall_correct: bool,
) -> str:
    """Build the retrospective self-assessment prompt.

    Delivered after the model has solved the problem. The model is told
    which steps were correct/incorrect but NOT given the correct answers.

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

    return f"""Here is your result: You got {correct_count} out of {n} steps correct.
Specifically:
{step_lines}
Your final answer was {overall_str}.

Now reflect on your performance:

1. HARDEST STEP: Which step actually gave you the most trouble \
during your reasoning? Give just the number.

2. PER-STEP SELF-ASSESSMENT: For each step, categorize your \
experience as exactly one of:
   - "confident and correct"
   - "confident but wrong"
   - "uncertain and correct"
   - "uncertain and wrong"

   Format as:
   Step 1: [label]
   Step 2: [label]
   ... etc.

3. COUNTERFACTUAL: Name one specific thing you considered doing \
differently but chose not to. What was the alternative approach, \
and why did you reject it?"""
