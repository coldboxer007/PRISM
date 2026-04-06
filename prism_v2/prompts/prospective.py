"""Dimension 1: Prospective Metacognitive Report prompt template."""


def build_prospective_prompt(problem_statement: str, num_steps: int) -> str:
    """Build the prospective metacognitive assessment prompt.

    This prompt is delivered BEFORE the model solves the problem.
    The model sees the problem statement and is asked to reflect before solving.
    """
    return f"""I'm going to ask you to solve a multi-step math problem. Before you \
solve it, I want you to assess your own capabilities on this specific problem.

Here is the problem:
{problem_statement}

This problem has {num_steps} steps. Before solving it, answer these three questions:

1. WEAKEST STEP: Which step number (1 through {num_steps}) do you predict \
will be the most difficult or error-prone for you? Give just the number.

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

Answer ONLY these three questions. Do not solve the problem yet."""
