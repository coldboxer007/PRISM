"""
PRISM v2.1 Validation Suite

Automated checks for:
  1. Ground-truth math correctness across all problem types
  2. Parser extraction on hardcoded test strings
  3. Scoring functions on synthetic data
  4. Difficulty distribution coverage

Run:  python -m prism_v2.validate
"""

import sys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_pass = 0
_fail = 0


def check(condition: bool, msg: str):
    global _pass, _fail
    if condition:
        _pass += 1
    else:
        _fail += 1
        print(f"  FAIL: {msg}")


# ---------------------------------------------------------------------------
# 1. Ground-truth math verification
# ---------------------------------------------------------------------------


def validate_ground_truth():
    print("--- Ground-truth math ---")
    from prism_v2.problems.generator import (
        generate_type_a_l1,
        generate_type_b_l1,
        generate_type_c_l1,
        generate_type_a_l2,
        generate_type_b_l2,
        generate_type_c_l2,
        _zeta_mul,
        _zeta_add,
        _zeta_sub,
        _star_add,
    )

    # Type-A L1: verify elimination steps
    p = generate_type_a_l1("test_a", "easy", seed=42)
    check(len(p.ground_truth_steps) == 5, "Type-A L1 should have 5 ground-truth steps")
    check(p.num_steps == 5, "Type-A L1 num_steps should be 5")
    # Final answer should contain x, y, z
    check("x =" in p.ground_truth_final, "Type-A L1 final should contain x =")

    # Type-B L1: subtotal = price * quantity
    p = generate_type_b_l1("test_b", "easy", seed=42)
    price = p.difficulty_metadata["price_per_unit"]
    units = p.difficulty_metadata["units"]
    expected_subtotal = price * units
    check(
        p.ground_truth_steps[0] == str(expected_subtotal),
        f"Type-B L1 step 1 subtotal: got {p.ground_truth_steps[0]}, expected {expected_subtotal}",
    )
    # Per-unit cost (step 5) should be total / units
    total = float(p.ground_truth_steps[3])
    expected_per_unit = round(total / units, 2)
    check(
        float(p.ground_truth_steps[4]) == expected_per_unit,
        f"Type-B L1 per-unit: got {p.ground_truth_steps[4]}, expected {expected_per_unit}",
    )

    # Type-C L1: modular chain
    p = generate_type_c_l1("test_c", "medium", seed=42)
    mod = p.difficulty_metadata["modulus"]
    ops = p.difficulty_metadata["operands"]
    a, b, c, d, e = ops
    s1 = a % mod
    s2 = (s1 + b) % mod
    s3 = (s2 * c) % mod
    s4 = (s3 + d) % mod
    s5 = pow(s4, e, mod)
    check(
        p.ground_truth_steps == [str(s1), str(s2), str(s3), str(s4), str(s5)],
        "Type-C L1 ground truth mismatch",
    )

    # Type-A L2: verify Zeta system ground truth
    p = generate_type_a_l2("test_a2", "easy", seed=42)
    check(len(p.ground_truth_steps) == 5, "Type-A L2 should have 5 ground-truth steps")
    check("x =" in p.ground_truth_final, "Type-A L2 final should contain x =")
    # The problem statement should reference Zeta-multiplication
    check("Zeta" in p.problem_statement, "Type-A L2 should mention Zeta operators")
    # It should be a system-solving task (structural parity with L1)
    check(
        "Solve" in p.problem_statement or "solve" in p.problem_statement,
        "Type-A L2 should ask model to solve (not just compute)",
    )

    # Type-B L2: verify Zeta operator math
    p = generate_type_b_l2("test_b2", "easy", seed=42)
    price = p.difficulty_metadata["price_per_unit"]
    units = p.difficulty_metadata["units"]
    subtotal = _zeta_mul(price, units)
    check(
        p.ground_truth_steps[0] == str(subtotal),
        f"Type-B L2 zeta subtotal: got {p.ground_truth_steps[0]}, expected {subtotal}",
    )

    # Type-C L2: verify Star-add math
    p = generate_type_c_l2("test_c2", "easy", seed=42)
    mod = p.difficulty_metadata["modulus"]
    ops = p.difficulty_metadata["operands"]
    a, b, c, d, e = ops
    s1 = a % mod
    s2 = _star_add(s1, b) % mod
    s3 = (s2 * c) % mod
    s4 = _star_add(s3, d) % mod
    s5 = pow(s4, e, mod)
    check(
        p.ground_truth_steps == [str(s1), str(s2), str(s3), str(s4), str(s5)],
        "Type-C L2 ground truth mismatch",
    )

    print(f"  Ground-truth checks complete")


# ---------------------------------------------------------------------------
# 2. Parser tests
# ---------------------------------------------------------------------------


def validate_parsers():
    print("--- Parser extraction ---")
    from prism_v2.scoring.confidence_parser import (
        parse_prospective,
        parse_retrospective,
    )
    from prism_v2.scoring.step_scorer import (
        extract_step_answers,
        compare_answers,
        score_steps,
    )

    # Prospective parser
    sample_d1 = (
        "WEAKEST STEP: 3\n"
        "Step 1: probably right\n"
        "Step 2: definitely right\n"
        "Step 3: probably wrong\n"
        "Step 4: uncertain\n"
        "Step 5: definitely right\n"
        "BET: $70 on correct, $30 on wrong"
    )
    d1 = parse_prospective(sample_d1, 5)
    check(d1.predicted_weakest_step == 3, "D1 weakest step should be 3")
    check(
        d1.confidence_vector == [4, 5, 2, 3, 5],
        f"D1 confidence vector: {d1.confidence_vector}",
    )
    check(
        abs(d1.bet_fraction_correct - 0.7) < 0.01,
        f"D1 bet fraction: {d1.bet_fraction_correct}",
    )
    check(
        len(d1.parse_errors) == 0, f"D1 should have no parse errors: {d1.parse_errors}"
    )

    # Retrospective parser
    sample_d3 = (
        "1. HARDEST STEP: 2\n"
        "2. SELF-ASSESSMENT:\n"
        "Step 1: confident and correct\n"
        "Step 2: uncertain and wrong\n"
        "Step 3: confident and correct\n"
        "Step 4: uncertain and correct\n"
        "Step 5: confident but wrong\n"
        "3. COUNTERFACTUAL: I could have used substitution instead of elimination."
    )
    d3 = parse_retrospective(sample_d3, 5)
    check(d3.reported_hardest_step == 2, f"D3 hardest step: {d3.reported_hardest_step}")
    check(
        d3.self_assessment[0] == "confident and correct",
        f"D3 assessment[0]: {d3.self_assessment[0]}",
    )
    check(
        d3.self_assessment[1] == "uncertain and wrong",
        f"D3 assessment[1]: {d3.self_assessment[1]}",
    )
    check(len(d3.counterfactual_text) > 0, "D3 counterfactual should be non-empty")

    # Empty counterfactual
    sample_no_cf = "1. HARDEST STEP: 1\n2. Step 1: uncertain and wrong\n"
    d3_no_cf = parse_retrospective(sample_no_cf, 1)
    check(
        d3_no_cf.counterfactual_text == "",
        "D3 without COUNTERFACTUAL header should be empty",
    )

    # Step scorer
    sample_solve = "Step 1: 2 + 3 = 5\nStep 2: 5 * 4 = 20\nStep 3: 20 - 7 = 13\n"
    answers = extract_step_answers(sample_solve, 3)
    check(answers[0] == "5", f"Step 1 answer: '{answers[0]}'")
    check(answers[1] == "20", f"Step 2 answer: '{answers[1]}'")
    check(answers[2] == "13", f"Step 3 answer: '{answers[2]}'")

    # compare_answers
    check(compare_answers("5", "5"), "5 == 5")
    check(compare_answers("5.0", "5"), "5.0 == 5")
    check(compare_answers("5.001", "5", tolerance=0.01), "5.001 ~= 5 within 1%")
    check(not compare_answers("6", "5"), "6 != 5")
    check(compare_answers("1/2", "0.5"), "1/2 == 0.5")

    # score_steps
    results = score_steps(["5", "20", "13"], ["5", "20", "14"])
    check(results == [True, True, False], f"score_steps: {results}")

    print(f"  Parser checks complete")


# ---------------------------------------------------------------------------
# 3. Scoring function tests
# ---------------------------------------------------------------------------


def validate_scoring():
    print("--- Scoring functions ---")
    from prism_v2.scoring.metrics import (
        compute_auroc,
        compute_spearman_rho,
        compute_step_accuracy_score,
        compute_retro_accuracy,
        compute_location_consistency,
        compute_confidence_consistency,
        compute_coherence_composite,
        compute_adaptive_calibration,
        compute_novelty_robustness,
    )

    # AUROC: perfect calibration
    auroc = compute_auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0])
    check(auroc == 1.0, f"Perfect AUROC should be 1.0, got {auroc}")

    # AUROC: chance (all same class)
    auroc_chance = compute_auroc([0.5, 0.6], [1, 1])
    check(auroc_chance == 0.5, f"All-positive AUROC should be 0.5, got {auroc_chance}")

    # AUROC: anti-calibration
    auroc_anti = compute_auroc([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0])
    check(auroc_anti == 0.0, f"Anti-calibrated AUROC should be 0.0, got {auroc_anti}")

    # Spearman: perfect positive
    rho = compute_spearman_rho([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    check(
        rho is not None and abs(rho - 1.0) < 0.01,
        f"Perfect Spearman should be 1.0, got {rho}",
    )

    # Spearman: perfect negative
    rho_neg = compute_spearman_rho([1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    check(
        rho_neg is not None and abs(rho_neg + 1.0) < 0.01,
        f"Negative Spearman: {rho_neg}",
    )

    # Spearman: zero variance
    rho_none = compute_spearman_rho([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
    check(rho_none is None, "Zero-variance Spearman should be None")

    # Step accuracy score
    score = compute_step_accuracy_score([1.0, -1.0, 0.0])
    check(abs(score - 0.5) < 0.01, f"Mean rho=0 -> score=0.5, got {score}")

    # Retro accuracy: all correct
    retro = compute_retro_accuracy(
        [["confident and correct", "uncertain and wrong"]],
        [[True, False]],
        [[5, 2]],
    )
    check(abs(retro - 1.0) < 0.01, f"Perfect retro should be 1.0, got {retro}")

    # Location consistency
    loc = compute_location_consistency([1, 2, 3], [1, 2, 4])
    check(abs(loc - 2 / 3) < 0.01, f"Location consistency: {loc}")

    # Coherence composite
    composite = compute_coherence_composite(0.8, 0.6, 0.4)
    expected = 0.3 * 0.8 + 0.4 * 0.6 + 0.3 * 0.4
    check(abs(composite - expected) < 0.001, f"Composite: {composite}")

    # Novelty robustness
    nr = compute_novelty_robustness(0.8, 0.6)
    check(abs(nr - 0.75) < 0.01, f"Novelty robustness: {nr}")
    nr_cap = compute_novelty_robustness(0.6, 0.8)
    check(abs(nr_cap - 1.0) < 0.01, f"Novelty robustness capped: {nr_cap}")
    nr_zero = compute_novelty_robustness(0.0, 0.5)
    check(nr_zero == 0.0, f"Novelty robustness L1=0: {nr_zero}")

    print(f"  Scoring checks complete")


# ---------------------------------------------------------------------------
# 4. Difficulty distribution
# ---------------------------------------------------------------------------


def validate_difficulty_distribution():
    print("--- Difficulty distribution ---")
    from prism_v2.problems.generator import generate_problem_set

    problems = generate_problem_set(novelty_level=1, base_seed=42, num_main=10)
    # Filter to main problems only
    main = [p for p in problems if "feedback" not in p.id]

    types_seen = set(p.problem_type for p in main)
    diffs_seen = set(p.difficulty for p in main)

    check(types_seen == {"A", "B", "C"}, f"Should have all 3 types, got {types_seen}")
    check(
        diffs_seen == {"easy", "medium", "hard"},
        f"Should have all 3 diffs, got {diffs_seen}",
    )

    # Verify decoupled: not every Type A is 'easy'
    type_a = [p for p in main if p.problem_type == "A"]
    a_diffs = set(p.difficulty for p in type_a)
    check(
        len(a_diffs) > 1,
        f"Type A should have varied difficulties (P3 fix), got {a_diffs}",
    )

    print(f"  Distribution checks complete")


# ---------------------------------------------------------------------------
# 5. Pipeline cache
# ---------------------------------------------------------------------------


def validate_pipeline_cache():
    print("--- Pipeline score cache ---")
    from prism_v2.pipeline import PrismPipeline

    pipeline = PrismPipeline([], [])
    check(
        hasattr(pipeline, "_task_score_cache"), "Pipeline should have _task_score_cache"
    )
    check(isinstance(pipeline._task_score_cache, dict), "Cache should be a dict")

    print(f"  Cache checks complete")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    global _pass, _fail

    print("PRISM v2.1 Validation Suite")
    print("=" * 40)

    validate_ground_truth()
    validate_parsers()
    validate_scoring()
    validate_difficulty_distribution()
    validate_pipeline_cache()

    print("=" * 40)
    print(f"Results: {_pass} passed, {_fail} failed")

    if _fail > 0:
        sys.exit(1)
    else:
        print("All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
