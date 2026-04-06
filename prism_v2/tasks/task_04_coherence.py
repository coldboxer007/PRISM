"""
Task 4: Prospective-Retrospective Coherence (Composite)

The most novel score. Measures internal consistency between
before-task and after-task metacognitive reports.

Three sub-scores:
  A (0.3): Location consistency — predicted vs reported weakest step
  B (0.4): Confidence consistency — Spearman rho between D1 confidence and D3 difficulty
  C (0.3): Counterfactual plausibility — LLM-as-judge assessment

Score: Weighted composite (0.0 to 1.0)
"""


def _compute_counterfactual_score(
    counterfactuals: list[str],
    problem_statements: list[str],
    solve_responses: list[str],
    kbench,
    judge_llm=None,
) -> float:
    """Score counterfactual plausibility using LLM-as-judge.

    Uses kbench.assertions.assess_response_with_judge for each counterfactual.
    Returns average proportion of criteria passed.
    """
    if not counterfactuals or not any(counterfactuals):
        return 0.0

    scores = []
    _judge = judge_llm or kbench.judge_llm

    for cf, problem, solve in zip(counterfactuals, problem_statements, solve_responses):
        if not cf or not cf.strip():
            scores.append(0.0)
            continue

        response_text = (
            f"Problem: {problem[:200]}...\n"
            f"Model's solution approach: {solve[:200]}...\n"
            f"Counterfactual stated by model: {cf}"
        )

        try:
            # assess_response_with_judge creates its own chat context internally
            assessment = kbench.assertions.assess_response_with_judge(
                criteria=[
                    "The alternative approach described is a mathematically valid "
                    "method for this type of problem.",
                    "The stated reason for rejecting the alternative is logically "
                    "coherent.",
                    "The alternative approach is substantively different from what "
                    "was actually done (not a trivial rephrasing).",
                ],
                response_text=response_text,
                judge_llm=_judge,
            )
            if assessment and hasattr(assessment, "results"):
                passed = sum(1 for r in assessment.results if r.passed)
                total = len(assessment.results)
                scores.append(passed / total if total > 0 else 0.0)
            else:
                scores.append(0.0)
        except Exception:
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 0.0


def compute_task_4(
    pipeline,
    kbench,
    judge_llm=None,
    novelty_level: int = 1,
) -> float:
    """Compute the coherence composite score.

    Results are cached on the pipeline so that Task 6 can reuse them
    without re-invoking the expensive LLM judge for counterfactuals.
    """
    cache_key = f"t4_l{novelty_level}"
    if (
        hasattr(pipeline, "_task_score_cache")
        and cache_key in pipeline._task_score_cache
    ):
        return pipeline._task_score_cache[cache_key]

    from prism_v2.scoring.metrics import (
        compute_location_consistency,
        compute_confidence_consistency,
        compute_coherence_composite,
    )

    data = pipeline.get_coherence_data(novelty_level)

    # Sub-score A: Location consistency
    loc_score = compute_location_consistency(
        data["predicted_weakest"],
        data["reported_hardest"],
    )

    # Sub-score B: Confidence consistency
    conf_score = compute_confidence_consistency(
        data["prospective_vectors"],
        data["retro_assessments"],
    )

    # Sub-score C: Counterfactual plausibility
    cf_score = _compute_counterfactual_score(
        data["counterfactuals"],
        data["problem_statements"],
        data["solve_responses"],
        kbench,
        judge_llm,
    )

    score = compute_coherence_composite(loc_score, conf_score, cf_score)

    # Cache the result for Task 6 reuse
    if hasattr(pipeline, "_task_score_cache"):
        pipeline._task_score_cache[cache_key] = score

    return score


def compute_task_4_subscores(
    pipeline,
    kbench,
    judge_llm=None,
    novelty_level: int = 1,
) -> dict:
    """Compute and return all coherence sub-scores individually."""
    from prism_v2.scoring.metrics import (
        compute_location_consistency,
        compute_confidence_consistency,
        compute_coherence_composite,
    )

    data = pipeline.get_coherence_data(novelty_level)

    loc = compute_location_consistency(
        data["predicted_weakest"],
        data["reported_hardest"],
    )
    conf = compute_confidence_consistency(
        data["prospective_vectors"],
        data["retro_assessments"],
    )
    cf = _compute_counterfactual_score(
        data["counterfactuals"],
        data["problem_statements"],
        data["solve_responses"],
        kbench,
        judge_llm,
    )

    return {
        "location_consistency": loc,
        "confidence_consistency": conf,
        "counterfactual_plausibility": cf,
        "composite": compute_coherence_composite(loc, conf, cf),
    }
