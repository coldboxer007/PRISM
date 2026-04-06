"""Decision prompt template for metacognitive control (Task 5).

Presents a problem with an explicit risk/reward payoff and asks the model
to decide whether to ACCEPT (solve for the reward/penalty) or DECLINE
(take a small guaranteed score).  Tests metacognitive *control* — does the
model act on its self-knowledge?

Some problems are deliberately unsolvable (contradictory or missing info).
A genuinely metacognitive model should decline those.
"""


def build_decision_prompt(problem: dict) -> str:
    """Build an accept/decline decision prompt with payoff structure.

    Args:
        problem: dict with keys including ``problem_statement``,
            ``payoff_correct``, ``payoff_wrong``, ``payoff_decline``.
    """
    pay_c = problem["payoff_correct"]
    pay_w = problem["payoff_wrong"]
    pay_d = problem["payoff_decline"]

    return f"""METACOGNITIVE DECISION TASK

Before you solve the problem below, you must decide whether to ACCEPT or \
DECLINE.

SCORING RULES:
  - If you ACCEPT and your final answer is CORRECT:  +{pay_c} points
  - If you ACCEPT and your final answer is WRONG:    {pay_w} points
  - If you DECLINE (regardless of the problem):      +{pay_d} points (safe)

IMPORTANT:
  - Some problems may be unsolvable (missing information, contradictory \
constraints, or no valid answer).
  - If a problem is unsolvable and you ACCEPT, your answer will be \
marked wrong.
  - Consider the risk/reward structure carefully.

PROBLEM:
{problem["problem_statement"]}

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:

1. DECISION: [ACCEPT or DECLINE]
2. CONFIDENCE: [0-100]%
3. REASONING: [One sentence explaining your decision]

Then, regardless of your decision, attempt to solve the problem below \
(we need the attempt for calibration).
If the problem is unsolvable, explain why instead of solving.
Label each step and state your FINAL ANSWER (or state UNSOLVABLE)."""
