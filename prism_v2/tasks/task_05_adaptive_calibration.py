"""
Task 5: Metacognitive Control (Accept/Decline Utility)

Measures whether the model acts on its self-knowledge by choosing to
ACCEPT or DECLINE problems based on an explicit payoff structure.

Replaces the earlier adaptive-calibration (confidence drift) task.

Score: Normalized utility ratio in [0, 1].
"""


def compute_task_5(pipeline, novelty_level: int = 1) -> float:
    """Compute metacognitive control score from decision problems.

    Retrieves accept/decline decisions and payoff data from the pipeline,
    then scores using the utility-ratio metric.
    """
    from prism_v2.scoring.metrics import compute_metacognitive_control

    data = pipeline.get_decision_data(novelty_level)

    if not data["decisions"]:
        return 0.0

    return compute_metacognitive_control(
        decisions=data["decisions"],
        is_solvable=data["is_solvable"],
        model_correct=data["model_correct"],
        payoff_correct=data["payoff_correct"],
        payoff_wrong=data["payoff_wrong"],
        payoff_decline=data["payoff_decline"],
    )


def compute_task_5_details(pipeline, novelty_level: int = 1) -> dict:
    """Compute metacognitive control with per-problem breakdown."""
    from prism_v2.scoring.metrics import compute_metacognitive_control

    data = pipeline.get_decision_data(novelty_level)

    if not data["decisions"]:
        return {"score": 0.0, "num_problems": 0, "decisions": []}

    score = compute_metacognitive_control(
        decisions=data["decisions"],
        is_solvable=data["is_solvable"],
        model_correct=data["model_correct"],
        payoff_correct=data["payoff_correct"],
        payoff_wrong=data["payoff_wrong"],
        payoff_decline=data["payoff_decline"],
    )

    results = pipeline.get_decision_results(novelty_level)
    per_problem = []
    for r in results:
        if r.decision == "accept":
            actual_payoff = r.payoff_correct if r.model_correct else r.payoff_wrong
        else:
            actual_payoff = r.payoff_decline
        per_problem.append(
            {
                "problem_id": r.problem_id,
                "is_solvable": r.is_solvable,
                "decision": r.decision,
                "confidence": r.confidence,
                "model_correct": r.model_correct,
                "actual_payoff": actual_payoff,
            }
        )

    return {
        "score": score,
        "num_problems": len(results),
        "decisions": per_problem,
    }
