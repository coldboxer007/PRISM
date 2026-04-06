"""Dimension 2: Performance (Problem Solving) prompt template."""


def build_solve_prompt() -> str:
    """Build the problem-solving prompt.

    Delivered as a follow-up message in the same conversation,
    so the model already has the problem statement and its D1 assessment.
    """
    return """Now solve the problem. Show your work clearly for each step. \
Label each step (Step 1, Step 2, etc.) and clearly state each \
intermediate answer.

After all steps, state your FINAL ANSWER clearly."""
