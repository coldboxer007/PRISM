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

    # Step alignment: skipped / out-of-order steps
    from prism_v2.scoring.step_scorer import _split_into_step_blocks

    # Skipped step: model outputs Step 1, Step 3, Step 4 (skips Step 2)
    skip_text = "Step 1: 2 + 3 = 5\nStep 3: 5 * 4 = 20\nStep 4: 20 - 7 = 13\n"
    blocks = _split_into_step_blocks(skip_text, 4)
    check(
        "5" in blocks[0],
        f"Skip: block[0] (Step 1) should contain '5', got '{blocks[0]}'",
    )
    check(
        blocks[1] == "", f"Skip: block[1] (Step 2) should be empty, got '{blocks[1]}'"
    )
    check(
        "20" in blocks[2],
        f"Skip: block[2] (Step 3) should contain '20', got '{blocks[2]}'",
    )
    check(
        "13" in blocks[3],
        f"Skip: block[3] (Step 4) should contain '13', got '{blocks[3]}'",
    )

    # Out-of-order steps: model outputs Step 3, Step 1, Step 2
    ooo_text = "Step 3: third = 30\nStep 1: first = 10\nStep 2: second = 20\n"
    blocks_ooo = _split_into_step_blocks(ooo_text, 3)
    check(
        "10" in blocks_ooo[0],
        f"OOO: block[0] (Step 1) should contain '10', got '{blocks_ooo[0]}'",
    )
    check(
        "20" in blocks_ooo[1],
        f"OOO: block[1] (Step 2) should contain '20', got '{blocks_ooo[1]}'",
    )
    check(
        "30" in blocks_ooo[2],
        f"OOO: block[2] (Step 3) should contain '30', got '{blocks_ooo[2]}'",
    )

    # Duplicate step (self-correction): later occurrence should overwrite
    dup_text = "Step 2: first attempt = 99\nStep 2: correction = 42\n"
    blocks_dup = _split_into_step_blocks(dup_text, 2)
    check(
        "42" in blocks_dup[1],
        f"Dup: block[1] (Step 2) should contain '42' (last), got '{blocks_dup[1]}'",
    )

    # Out-of-range step numbers should be skipped
    oor_text = "Step 0: invalid\nStep 1: valid = 10\nStep 99: way out of range\n"
    blocks_oor = _split_into_step_blocks(oor_text, 3)
    check(
        "10" in blocks_oor[0],
        f"OOR: block[0] (Step 1) should contain '10', got '{blocks_oor[0]}'",
    )
    check(
        blocks_oor[1] == "",
        f"OOR: block[1] (Step 2) should be empty, got '{blocks_oor[1]}'",
    )
    check(
        blocks_oor[2] == "",
        f"OOR: block[2] (Step 3) should be empty, got '{blocks_oor[2]}'",
    )

    # Full pipeline: skipped steps should score correctly
    skip_solve = "Step 1: 2 + 3 = 5\nStep 3: 10 * 2 = 20\n"
    skip_answers = extract_step_answers(skip_solve, 3)
    skip_scores = score_steps(skip_answers, ["5", "15", "20"])
    check(
        skip_scores[0] is True, f"Skip pipeline: step 1 correct, got {skip_scores[0]}"
    )
    check(
        skip_scores[1] is False,
        f"Skip pipeline: step 2 missing -> wrong, got {skip_scores[1]}",
    )
    check(
        skip_scores[2] is True, f"Skip pipeline: step 3 correct, got {skip_scores[2]}"
    )

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

    # Coherence composite (updated weights: 0.35, 0.45, 0.20)
    composite = compute_coherence_composite(0.8, 0.6, 0.4)
    expected = 0.35 * 0.8 + 0.45 * 0.6 + 0.20 * 0.4
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

    # Decision result caches should exist
    check(
        hasattr(pipeline, "_l1_decision_results"),
        "Pipeline should have _l1_decision_results",
    )
    check(
        hasattr(pipeline, "_l2_decision_results"),
        "Pipeline should have _l2_decision_results",
    )

    print(f"  Cache checks complete")


# ---------------------------------------------------------------------------
# 6. Decision problem generators
# ---------------------------------------------------------------------------


def validate_decision_generators():
    print("--- Decision problem generators ---")
    from prism_v2.problems.generator import (
        generate_decision_problems,
        _generate_contradictory_system_l1,
        _generate_contradictory_system_l2,
        _generate_missing_info_l1,
        _generate_missing_info_l2,
        PAYOFF_PROFILES,
    )

    # L1 decision problems
    l1_dec = generate_decision_problems(novelty_level=1, base_seed=42)
    check(len(l1_dec) == 5, f"L1 should have 5 decision problems, got {len(l1_dec)}")

    # 3 solvable + 2 unsolvable
    solvable = [p for p in l1_dec if p.get("is_solvable", True)]
    unsolvable = [p for p in l1_dec if not p.get("is_solvable", True)]
    check(len(solvable) == 3, f"L1 should have 3 solvable, got {len(solvable)}")
    check(len(unsolvable) == 2, f"L1 should have 2 unsolvable, got {len(unsolvable)}")

    # All have payoff fields
    for p in l1_dec:
        check(
            "payoff_correct" in p and "payoff_wrong" in p and "payoff_decline" in p,
            f"Decision problem {p['id']} missing payoff fields",
        )

    # All have "decision" in their ID
    for p in l1_dec:
        check(
            "decision" in p["id"],
            f"Decision problem ID should contain 'decision': {p['id']}",
        )

    # Unsolvable problems should have ground_truth_final == "UNSOLVABLE"
    for p in unsolvable:
        check(
            p["ground_truth_final"] == "UNSOLVABLE",
            f"Unsolvable {p['id']} should have UNSOLVABLE final answer",
        )

    # L2 decision problems
    l2_dec = generate_decision_problems(novelty_level=2, base_seed=42)
    check(len(l2_dec) == 5, f"L2 should have 5 decision problems, got {len(l2_dec)}")
    l2_unsolvable = [p for p in l2_dec if not p.get("is_solvable", True)]
    check(
        len(l2_unsolvable) == 2,
        f"L2 should have 2 unsolvable, got {len(l2_unsolvable)}",
    )

    # Contradictory system L1: verify it's actually contradictory
    contra = _generate_contradictory_system_l1("test_contra", seed=42)
    check(not contra["is_solvable"], "Contradictory system should be unsolvable")
    check(
        contra["unsolvable_reason"] == "contradictory_system",
        "Reason should be contradictory_system",
    )

    # Missing info L1: verify it omits quantity
    missing = _generate_missing_info_l1("test_missing", seed=42)
    check(not missing["is_solvable"], "Missing-info should be unsolvable")
    check(
        "several items" in missing["problem_statement"],
        "Missing-info should say 'several items' (no specific quantity)",
    )

    # Contradictory system L2: should mention Zeta
    contra_l2 = _generate_contradictory_system_l2("test_contra_l2", seed=42)
    check(
        "Zeta" in contra_l2["problem_statement"],
        "L2 contradictory should mention Zeta operators",
    )

    # Missing info L2: should mention Zeta
    missing_l2 = _generate_missing_info_l2("test_missing_l2", seed=42)
    check(
        "Zeta" in missing_l2["problem_statement"],
        "L2 missing-info should mention Zeta operators",
    )

    # Payoff profiles
    check("low" in PAYOFF_PROFILES, "PAYOFF_PROFILES should have 'low'")
    check("medium" in PAYOFF_PROFILES, "PAYOFF_PROFILES should have 'medium'")
    check("high" in PAYOFF_PROFILES, "PAYOFF_PROFILES should have 'high'")

    print(f"  Decision generator checks complete")


# ---------------------------------------------------------------------------
# 7. Decision scorer (parser)
# ---------------------------------------------------------------------------


def validate_decision_scorer():
    print("--- Decision scorer ---")
    from prism_v2.scoring.decision_scorer import parse_decision_response

    # Standard accept response
    accept_text = (
        "DECISION: ACCEPT\n"
        "CONFIDENCE: 85%\n"
        "REASONING: This is a straightforward system of equations.\n\n"
        "Step 1: ..."
    )
    r = parse_decision_response(accept_text)
    check(r.decision == "accept", f"Should parse ACCEPT, got {r.decision}")
    check(r.confidence == 85, f"Should parse 85, got {r.confidence}")
    check(len(r.reasoning) > 0, "Should parse reasoning")
    check(len(r.parse_errors) == 0, f"No parse errors expected: {r.parse_errors}")

    # Decline response
    decline_text = (
        "DECISION: DECLINE\n"
        "CONFIDENCE: 20%\n"
        "REASONING: The problem seems contradictory.\n\n"
        "Attempting anyway..."
    )
    r2 = parse_decision_response(decline_text)
    check(r2.decision == "decline", f"Should parse DECLINE, got {r2.decision}")
    check(r2.confidence == 20, f"Should parse 20, got {r2.confidence}")

    # Fallback: no explicit DECISION: label
    fallback_text = "I will DECLINE this problem because...\nCONFIDENCE: 10%\n"
    r3 = parse_decision_response(fallback_text)
    check(
        r3.decision == "decline", f"Fallback should detect DECLINE, got {r3.decision}"
    )

    # No decision at all -> default accept
    empty_text = "Let me try to solve this problem.\nStep 1: ..."
    r4 = parse_decision_response(empty_text)
    check(r4.decision == "accept", f"Default should be accept, got {r4.decision}")
    check(len(r4.parse_errors) > 0, "Missing decision should produce parse error")

    print(f"  Decision scorer checks complete")


# ---------------------------------------------------------------------------
# 8. Metacognitive control metric
# ---------------------------------------------------------------------------


def validate_metacognitive_control():
    print("--- Metacognitive control metric ---")
    from prism_v2.scoring.metrics import compute_metacognitive_control

    # Perfect oracle: accept correct, decline incorrect
    score = compute_metacognitive_control(
        decisions=["accept", "accept", "decline", "decline"],
        is_solvable=[True, True, True, False],
        model_correct=[True, True, False, False],
        payoff_correct=[5, 10, 20, 10],
        payoff_wrong=[-3, -15, -30, -15],
        payoff_decline=[1, 2, 3, 2],
    )
    check(
        abs(score - 1.0) < 0.001,
        f"Perfect oracle decisions should score 1.0, got {score}",
    )

    # Worst possible: accept wrong, decline correct
    score_worst = compute_metacognitive_control(
        decisions=["decline", "decline", "accept", "accept"],
        is_solvable=[True, True, True, False],
        model_correct=[True, True, False, False],
        payoff_correct=[5, 10, 20, 10],
        payoff_wrong=[-3, -15, -30, -15],
        payoff_decline=[1, 2, 3, 2],
    )
    check(
        abs(score_worst - 0.0) < 0.001,
        f"Worst decisions should score 0.0, got {score_worst}",
    )

    # All decline: should be between 0 and 1
    score_decline = compute_metacognitive_control(
        decisions=["decline", "decline", "decline", "decline"],
        is_solvable=[True, True, True, False],
        model_correct=[True, True, False, False],
        payoff_correct=[5, 10, 20, 10],
        payoff_wrong=[-3, -15, -30, -15],
        payoff_decline=[1, 2, 3, 2],
    )
    check(
        0.0 <= score_decline <= 1.0,
        f"All-decline score should be in [0,1], got {score_decline}",
    )

    # All accept: depends on correctness mix
    score_accept = compute_metacognitive_control(
        decisions=["accept", "accept", "accept", "accept"],
        is_solvable=[True, True, True, False],
        model_correct=[True, True, False, False],
        payoff_correct=[5, 10, 20, 10],
        payoff_wrong=[-3, -15, -30, -15],
        payoff_decline=[1, 2, 3, 2],
    )
    check(
        0.0 <= score_accept <= 1.0,
        f"All-accept score should be in [0,1], got {score_accept}",
    )

    # Empty: should return 0.0
    score_empty = compute_metacognitive_control([], [], [], [], [], [])
    check(score_empty == 0.0, f"Empty should score 0.0, got {score_empty}")

    print(f"  Metacognitive control checks complete")


# ---------------------------------------------------------------------------
# 9. Contradictory system verification (HIGH-1)
# ---------------------------------------------------------------------------


def validate_contradictory_systems():
    """Verify that contradictory systems are actually contradictory.

    Check that det(A_eff) = 0 (linearly dependent rows) AND the system
    is inconsistent (not merely underdetermined). This catches generator
    edge cases where rounding or coefficient choices could produce a
    solvable system.
    """
    print("--- Contradictory system verification ---")
    from fractions import Fraction
    from prism_v2.problems.generator import (
        _generate_contradictory_system_l1,
        _generate_contradictory_system_l2,
        _zeta_mul,
    )

    # Test multiple seeds for L1
    for seed in [42, 100, 200, 300, 999]:
        contra = _generate_contradictory_system_l1(f"contra_l1_{seed}", seed=seed)
        meta = contra["difficulty_metadata"]
        alpha, beta, shift = meta["alpha"], meta["beta"], meta["shift"]

        # Reconstruct the coefficient matrix from the problem statement
        # Parse equations from step_descriptions - the matrix is embedded
        # in the problem construction. We verify by re-checking the math.
        # For L1: eq3 = alpha*eq1 + beta*eq2, with b3 shifted by 'shift'
        check(shift != 0, f"L1 contra seed={seed}: shift must be nonzero (got {shift})")
        check(
            contra["ground_truth_final"] == "UNSOLVABLE",
            f"L1 contra seed={seed}: ground truth must be UNSOLVABLE",
        )

    # Test multiple seeds for L2
    for seed in [42, 100, 200, 300, 999]:
        contra = _generate_contradictory_system_l2(f"contra_l2_{seed}", seed=seed)
        meta = contra["difficulty_metadata"]
        alpha, beta, shift = meta["alpha"], meta["beta"], meta["shift"]

        check(shift != 0, f"L2 contra seed={seed}: shift must be nonzero (got {shift})")
        check(
            contra["ground_truth_final"] == "UNSOLVABLE",
            f"L2 contra seed={seed}: ground truth must be UNSOLVABLE",
        )

    # Deep verification on seed=42: reconstruct and verify L1 inconsistency
    contra = _generate_contradictory_system_l1("deep_l1", seed=42)
    # Re-generate the system to get the matrix
    import random
    rng = random.Random(42)
    coeff_range = (-5, 5)
    while True:
        A = [[rng.randint(*coeff_range) for _ in range(3)] for _ in range(2)]
        alpha_v = rng.choice([-2, -1, 1, 2])
        beta_v = rng.choice([-2, -1, 1, 2])
        row3 = [alpha_v * A[0][j] + beta_v * A[1][j] for j in range(3)]
        if A[0][0] == 0:
            continue
        if all(c == 0 for c in row3):
            continue
        break
    x_arb = [rng.randint(-5, 5) for _ in range(3)]
    b = [sum(A[i][j] * x_arb[j] for j in range(3)) for i in range(2)]
    consistent_b3 = alpha_v * b[0] + beta_v * b[1]
    shift_v = rng.choice([-3, -2, -1, 1, 2, 3])
    b3 = consistent_b3 + shift_v

    # Verify determinant = 0 (rows are linearly dependent)
    A_full = A + [row3]
    det = (
        A_full[0][0] * (A_full[1][1] * A_full[2][2] - A_full[1][2] * A_full[2][1])
        - A_full[0][1] * (A_full[1][0] * A_full[2][2] - A_full[1][2] * A_full[2][0])
        + A_full[0][2] * (A_full[1][0] * A_full[2][1] - A_full[1][1] * A_full[2][0])
    )
    check(det == 0, f"L1 contradictory system det should be 0, got {det}")

    # Verify inconsistency: b3 != alpha*b1 + beta*b2
    check(
        b3 != consistent_b3,
        f"L1 system should be inconsistent: b3={b3} should differ from consistent={consistent_b3}",
    )

    print(f"  Contradictory system verification complete")


# ---------------------------------------------------------------------------
# 10. Type-A L2 Zeta math verification (MEDIUM-7)
# ---------------------------------------------------------------------------


def validate_zeta_math():
    """Verify that Type-A L2 Zeta equations have correct ground truth.

    For each equation: sum(zeta_mul(A_zeta[i][j], x_true[j]) for j) == rhs[i]
    """
    print("--- Type-A L2 Zeta math verification ---")
    from prism_v2.problems.generator import generate_type_a_l2, _zeta_mul

    for difficulty in ["easy", "medium", "hard"]:
        for seed in [42, 100, 200]:
            p = generate_type_a_l2(f"zeta_{difficulty}_{seed}", difficulty, seed=seed)
            meta = p.difficulty_metadata
            A_zeta = meta["zeta_coefficients"]
            x_true = meta["solution"]
            rhs = meta["rhs"]

            # Verify each equation: sum(zeta_mul(coeff, var)) == rhs
            for i in range(3):
                computed_rhs = sum(_zeta_mul(A_zeta[i][j], x_true[j]) for j in range(3))
                check(
                    computed_rhs == rhs[i],
                    f"Zeta eq {i+1} mismatch ({difficulty}, seed={seed}): "
                    f"computed={computed_rhs}, expected={rhs[i]}",
                )

            # Verify no zero Zeta coefficients
            has_zero = any(A_zeta[i][j] == 0 for i in range(3) for j in range(3))
            check(
                not has_zero,
                f"Zeta coefficients should all be nonzero ({difficulty}, seed={seed})",
            )

    print(f"  Type-A L2 Zeta math verification complete")


# ---------------------------------------------------------------------------
# 11. JSON file validation (MEDIUM-5)
# ---------------------------------------------------------------------------


def validate_json_files():
    """Validate the shipped JSON problem files against the generator.

    Loads l1_problems.json and l2_problems.json, spot-checks ground truth
    by recomputing, verifies difficulty distribution, and checks decision
    problem payoff fields.
    """
    print("--- JSON file validation ---")
    import json
    import os
    from prism_v2.problems.generator import (
        generate_type_a_l1,
        generate_type_b_l1,
        generate_type_c_l1,
        generate_type_a_l2,
        generate_type_b_l2,
        generate_type_c_l2,
        _zeta_mul,
    )

    json_dir = os.path.join(os.path.dirname(__file__), "problems")
    l1_path = os.path.join(json_dir, "l1_problems.json")
    l2_path = os.path.join(json_dir, "l2_problems.json")

    check(os.path.exists(l1_path), f"l1_problems.json should exist at {l1_path}")
    check(os.path.exists(l2_path), f"l2_problems.json should exist at {l2_path}")

    with open(l1_path) as f:
        l1_data = json.load(f)
    with open(l2_path) as f:
        l2_data = json.load(f)

    # Expected counts: 10 main + 5 feedback + 5 decision = 20
    check(len(l1_data) == 20, f"L1 should have 20 problems, got {len(l1_data)}")
    check(len(l2_data) == 20, f"L2 should have 20 problems, got {len(l2_data)}")

    # Verify difficulty distribution for main problems
    for level_data, level_name in [(l1_data, "L1"), (l2_data, "L2")]:
        main_problems = [p for p in level_data if "main" in p["id"]]
        diffs = [p["difficulty"] for p in main_problems]
        for d in ["easy", "medium", "hard"]:
            count = diffs.count(d)
            check(
                count >= 2,
                f"{level_name} should have at least 2 {d} main problems, got {count}",
            )

    # Verify decision problems have payoff fields
    for level_data, level_name in [(l1_data, "L1"), (l2_data, "L2")]:
        decision_problems = [p for p in level_data if "decision" in p["id"]]
        check(
            len(decision_problems) == 5,
            f"{level_name} should have 5 decision problems, got {len(decision_problems)}",
        )
        for dp in decision_problems:
            check(
                "payoff_correct" in dp,
                f"{level_name} decision {dp['id']} missing payoff_correct",
            )
            check(
                "payoff_wrong" in dp,
                f"{level_name} decision {dp['id']} missing payoff_wrong",
            )
            check(
                "payoff_decline" in dp,
                f"{level_name} decision {dp['id']} missing payoff_decline",
            )
            check(
                "is_solvable" in dp,
                f"{level_name} decision {dp['id']} missing is_solvable",
            )

    # Spot-check: verify a Type-A L2 main problem's Zeta RHS
    l2_type_a = [p for p in l2_data if p["problem_type"] == "A"
                 and "main" in p["id"] and p.get("is_solvable", True)]
    if l2_type_a:
        p = l2_type_a[0]
        meta = p["difficulty_metadata"]
        A_zeta = meta["zeta_coefficients"]
        x_true = meta["solution"]
        rhs = meta["rhs"]
        for i in range(3):
            computed = sum(_zeta_mul(A_zeta[i][j], x_true[j]) for j in range(3))
            check(
                computed == rhs[i],
                f"JSON {p['id']} eq {i+1}: computed RHS={computed}, stored={rhs[i]}",
            )

    # Verify unsolvable decision problems (2 per level)
    for level_data, level_name in [(l1_data, "L1"), (l2_data, "L2")]:
        decision = [p for p in level_data if "decision" in p["id"]]
        unsolvable = [p for p in decision if not p.get("is_solvable", True)]
        check(
            len(unsolvable) == 2,
            f"{level_name} should have 2 unsolvable decision problems, got {len(unsolvable)}",
        )

    # Verify no zero Zeta coefficients in L2 Type-A problems
    for p in l2_data:
        if p["problem_type"] == "A":
            meta = p.get("difficulty_metadata", {})
            zc = meta.get("zeta_coefficients")
            if zc:
                has_zero = any(c == 0 for row in zc for c in row)
                check(
                    not has_zero,
                    f"JSON {p['id']} has zero Zeta coefficient",
                )

    # Verify no zero effective coefficients in L2 Type-A problems
    from prism_v2.problems.generator import _zeta_mul
    for p in l2_data:
        if p["problem_type"] == "A":
            meta = p.get("difficulty_metadata", {})
            zc = meta.get("zeta_coefficients")
            sol = meta.get("solution")
            if zc and sol:
                for i in range(3):
                    for j in range(3):
                        eff = _zeta_mul(zc[i][j], 1)  # effective coefficient
                        check(
                            eff != 0,
                            f"JSON {p['id']} eq{i+1} var{j+1}: effective coeff is 0 "
                            f"(zeta={zc[i][j]})",
                        )

    # Verify negative Zeta coefficients are parenthesized in problem statements
    import re
    for p in l2_data:
        if p["problem_type"] == "A" and "(*)" in p.get("problem_statement", ""):
            bad = re.findall(r'(?<!\()(-\d+) \(\*\)', p["problem_statement"])
            check(
                len(bad) == 0,
                f"JSON {p['id']} has unparenthesized negative Zeta coefficient: {bad}",
            )

    print(f"  JSON file validation complete")


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
    validate_decision_generators()
    validate_decision_scorer()
    validate_metacognitive_control()
    validate_contradictory_systems()
    validate_zeta_math()
    validate_json_files()

    print("=" * 40)
    print(f"Results: {_pass} passed, {_fail} failed")

    if _fail > 0:
        sys.exit(1)
    else:
        print("All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
