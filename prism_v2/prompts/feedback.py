"""Feedback round prompt template for adaptive calibration (Task 5)."""


def build_feedback_prompt(
    prev_step_results: list[bool],
    prev_confidence_labels: list[str],
    new_problem_statement: str,
    num_steps: int,
) -> str:
    """Build a combined prospective-assessment + solve prompt for feedback rounds.

    In feedback rounds, we combine the prospective report and solving into
    a single prompt to save API quota.

    Args:
        prev_step_results: per-step correctness from the previous round
        prev_confidence_labels: per-step confidence labels from the previous round
        new_problem_statement: the new problem to solve
        num_steps: number of steps in the new problem
    """
    feedback_lines = "\n".join(
        f'Step {i + 1}: {"CORRECT" if r else "INCORRECT"} — you predicted "{c}"'
        for i, (r, c) in enumerate(zip(prev_step_results, prev_confidence_labels))
    )

    return f"""In the previous problem, your per-step results were:
{feedback_lines}

Here is a new problem of similar type:
{new_problem_statement}

Given what you learned from the previous problem, provide your \
updated prospective assessment:

1. WEAKEST STEP: Which step will be hardest? (number only)
2. PER-STEP CONFIDENCE: For each step, rate your confidence using \
exactly one of these labels:
   - "definitely right"
   - "probably right"
   - "uncertain"
   - "probably wrong"
   - "definitely wrong"

   Format as:
   Step 1: [label]
   Step 2: [label]
   ... etc.

3. OVERALL BET: If you had to bet $100 on whether your final answer \
will be correct, how would you split it? State the amount you'd \
bet on CORRECT and the amount on INCORRECT. They must sum to $100.

Then solve the problem showing your work for each step.
Label each step (Step 1, Step 2, etc.) and clearly state each \
intermediate answer.

After all steps, state your FINAL ANSWER clearly."""
