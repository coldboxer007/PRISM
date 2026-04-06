"""
PRISM v2.1 Problem Generator

Generates multi-step mathematical problems with deterministic intermediate answers.
Supports both L1 (familiar/standard) and L2 (novel/custom operator) problem types.

Problem Types:
  A - Systems of Linear Equations (3 variables, 5 steps)
  B - Multi-Step Word Problems (5 steps)
  C - Modular Arithmetic / Number Theory (5 steps)
"""

import json
import random
from dataclasses import dataclass, field, asdict
from typing import Optional
from fractions import Fraction


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Problem:
    """A single multi-step problem with ground-truth intermediate answers."""

    id: str
    novelty_level: int  # 1 = familiar, 2 = novel
    problem_type: str  # "A", "B", or "C"
    difficulty: str  # "easy", "medium", "hard"
    problem_statement: str
    num_steps: int
    step_descriptions: list[str]
    ground_truth_steps: list[str]  # expected answer per step
    ground_truth_final: str
    difficulty_metadata: dict = field(default_factory=dict)
    operator_preamble: Optional[str] = None  # only for L2

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# L1 Type-A: Systems of linear equations  (3 vars, 5 steps)
# ---------------------------------------------------------------------------


def _generate_system_coefficients(
    coeff_range: tuple[int, int],
    sol_range: tuple[int, int],
    rng: random.Random,
) -> tuple[list[list[int]], list[int], list[int]]:
    """Generate a 3x3 coefficient matrix A, solution vector x, constant vector b=Ax.
    Retries until det(A) != 0 and A[0][0] != 0 (needed for elimination)."""
    while True:
        A = [[rng.randint(*coeff_range) for _ in range(3)] for _ in range(3)]
        # Ensure A[0][0] is non-zero (needed for step 1 of elimination)
        if A[0][0] == 0:
            continue
        det = (
            A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
            - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
            + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
        )
        if det != 0:
            break
    x = [rng.randint(*sol_range) for _ in range(3)]
    b = [sum(A[i][j] * x[j] for j in range(3)) for i in range(3)]
    return A, x, b


def _format_equation(coeffs: list[int], rhs: int, var_names: list[str] = None) -> str:
    """Format a single linear equation as a readable string."""
    if var_names is None:
        var_names = ["x", "y", "z"]
    parts = []
    for i, c in enumerate(coeffs):
        if c == 0:
            continue
        sign = "+" if c > 0 and parts else ("-" if c < 0 else "")
        ac = abs(c)
        coeff_str = "" if ac == 1 else str(ac)
        parts.append(f"{sign}{' ' if parts and sign else ''}{coeff_str}{var_names[i]}")
    if not parts:
        parts = ["0"]
    return " ".join(parts) + f" = {rhs}"


def _solve_system_steps(
    A: list[list[int]], b: list[int], x_true: list[int]
) -> tuple[list[str], list[str], list[str], str]:
    """Solve a 3x3 system step-by-step using elimination.

    Instead of checking algebraic expressions (which LLMs format inconsistently),
    we restructure Type-A as a 5-step numeric computation that can be verified:
      Step 1: Compute the multiplier m21 = A[1][0]/A[0][0] (used to eliminate x from eq2)
      Step 2: Compute the new RHS of eq2 after elimination: b[1] - m21*b[0]
      Step 3: Repeat for eq3 (compute m31 and new RHS after eliminating x)
      Step 4: Solve for z (final variable)
      Step 5: Back-substitute to find y

    Returns (step_descriptions, step_answers, ground_truth_steps, final_answer).
    ground_truth_steps contains purely numeric values for reliable step scoring.
    """
    F = Fraction

    a00, a01, a02 = [F(c) for c in A[0]]
    a10, a11, a12 = [F(c) for c in A[1]]
    a20, a21, a22 = [F(c) for c in A[2]]
    b0, b1, b2 = F(b[0]), F(b[1]), F(b[2])

    # Step 1: Multiplier to eliminate x from equation 2
    m21 = a10 / a00
    m21_f = float(m21)
    desc_1 = f"Compute the multiplier m = A[1][0]/A[0][0] = {A[1][0]}/{A[0][0]}"
    ans_1 = f"m = {m21_f:.6g}"
    gt_1 = f"{m21_f:.6g}"

    # Step 2: New coefficient and RHS for eq2 after eliminating x
    # new_b1 = b1 - m21 * b0
    new_b1 = b1 - m21 * b0
    new_a11 = a11 - m21 * a01
    new_a12 = a12 - m21 * a02
    desc_2 = f"Eliminate x from equation 2: new RHS = {float(b1):.6g} - {m21_f:.6g}*{float(b0):.6g}"
    ans_2 = f"new RHS = {float(new_b1):.6g}"
    gt_2 = f"{float(new_b1):.6g}"

    # Step 3: Multiplier and new RHS for eq3 after eliminating x
    m31 = a20 / a00
    new_b2 = b2 - m31 * b0
    new_a21 = a21 - m31 * a01
    new_a22 = a22 - m31 * a02
    desc_3 = f"Eliminate x from equation 3: new RHS = {float(b2):.6g} - {float(m31):.6g}*{float(b0):.6g}"
    ans_3 = f"new RHS = {float(new_b2):.6g}"
    gt_3 = f"{float(new_b2):.6g}"

    # Step 4: Solve for z (after eliminating y from the reduced system)
    z_val = F(x_true[2])
    desc_4 = "Solve the reduced 2x2 system for z"
    ans_4 = f"z = {float(z_val):.6g}"
    gt_4 = str(x_true[2])

    # Step 5: Back-substitute to find y
    y_val = F(x_true[1])
    desc_5 = "Back-substitute z to find y"
    ans_5 = f"y = {float(y_val):.6g}"
    gt_5 = str(x_true[1])

    descs = [desc_1, desc_2, desc_3, desc_4, desc_5]
    answers = [ans_1, ans_2, ans_3, ans_4, ans_5]
    gt_steps = [gt_1, gt_2, gt_3, gt_4, gt_5]
    final = f"x = {x_true[0]}, y = {x_true[1]}, z = {x_true[2]}"

    return descs, answers, gt_steps, final


def generate_type_a_l1(
    problem_id: str,
    difficulty: str = "easy",
    seed: int = 42,
) -> Problem:
    """Generate an L1 (familiar) Type-A problem: system of 3 linear equations."""
    rng = random.Random(seed)

    ranges = {
        "easy": ((-3, 3), (-5, 5)),
        "medium": ((-5, 7), (-10, 10)),
        "hard": ((-10, 10), (-20, 20)),
    }
    coeff_range, sol_range = ranges[difficulty]

    A, x_true, b = _generate_system_coefficients(coeff_range, sol_range, rng)

    eqs = [_format_equation(A[i], b[i]) for i in range(3)]
    statement = (
        "Solve the following system of linear equations for x, y, and z "
        "using Gaussian elimination.\n"
        "Show your work for each step.\n\n"
        f"Equation 1: {eqs[0]}\n"
        f"Equation 2: {eqs[1]}\n"
        f"Equation 3: {eqs[2]}\n\n"
        "Steps:\n"
        f"Step 1: Compute the multiplier m = A[1][0]/A[0][0] = {A[1][0]}/{A[0][0]}\n"
        "Step 2: Eliminate x from equation 2 using the multiplier. State the new RHS.\n"
        f"Step 3: Eliminate x from equation 3 (multiplier = {A[2][0]}/{A[0][0]}). State the new RHS.\n"
        "Step 4: Solve the reduced 2x2 system for z.\n"
        "Step 5: Back-substitute z to find y."
    )

    descs, answers, gt_steps, final = _solve_system_steps(A, b, x_true)

    return Problem(
        id=problem_id,
        novelty_level=1,
        problem_type="A",
        difficulty=difficulty,
        problem_statement=statement,
        num_steps=5,
        step_descriptions=descs,
        ground_truth_steps=gt_steps,
        ground_truth_final=final,
        difficulty_metadata={
            "coefficient_range": list(coeff_range),
            "solution_range": list(sol_range),
            "has_fractions": any(
                A[i][j] != 0 and b[i] % A[i][j] != 0 for i in range(3) for j in range(3)
            ),
            "determinant": int(
                A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
                - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
                + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])
            ),
        },
    )


# ---------------------------------------------------------------------------
# L1 Type-B: Multi-step word problems (5 steps)
# ---------------------------------------------------------------------------


def generate_type_b_l1(
    problem_id: str,
    difficulty: str = "easy",
    seed: int = 42,
) -> Problem:
    """Generate an L1 Type-B problem: multi-step word problem."""
    rng = random.Random(seed)

    templates = {
        "easy": {
            "price_per_unit": rng.randint(3, 12),
            "units": rng.randint(5, 15),
            "tax_pct": rng.choice([5, 8, 10]),
            "discount_pct": rng.choice([10, 15, 20]),
            "shipping": rng.randint(3, 10),
        },
        "medium": {
            "price_per_unit": rng.randint(15, 50),
            "units": rng.randint(10, 30),
            "tax_pct": rng.choice([7, 9, 12, 15]),
            "discount_pct": rng.choice([5, 12, 18, 25]),
            "shipping": rng.randint(8, 25),
        },
        "hard": {
            "price_per_unit": rng.randint(25, 150),
            "units": rng.randint(15, 50),
            "tax_pct": rng.choice([6.5, 8.25, 11.5, 13.75]),
            "discount_pct": rng.choice([7.5, 13.5, 22, 27.5]),
            "shipping": rng.randint(12, 45),
        },
    }
    p = templates[difficulty]

    # Step 1: Compute subtotal
    subtotal = p["price_per_unit"] * p["units"]
    # Step 2: Apply discount
    discount_amount = round(subtotal * p["discount_pct"] / 100, 2)
    after_discount = round(subtotal - discount_amount, 2)
    # Step 3: Apply tax
    tax_amount = round(after_discount * p["tax_pct"] / 100, 2)
    after_tax = round(after_discount + tax_amount, 2)
    # Step 4: Add shipping
    total = round(after_tax + p["shipping"], 2)
    # Step 5: Per-unit final cost
    per_unit = round(total / p["units"], 2)

    statement = (
        f"A customer orders {p['units']} items at ${p['price_per_unit']} each. "
        f"They receive a {p['discount_pct']}% discount on the subtotal. "
        f"A sales tax of {p['tax_pct']}% is applied after the discount. "
        f"Finally, a flat shipping fee of ${p['shipping']} is added.\n\n"
        "Calculate each of the following, showing your work step by step:\n"
        "Step 1: Compute the subtotal (price x quantity)\n"
        "Step 2: Compute the discount amount and price after discount\n"
        "Step 3: Compute the tax amount and price after tax\n"
        "Step 4: Compute the total including shipping\n"
        "Step 5: Compute the cost per unit (total / quantity)"
    )

    return Problem(
        id=problem_id,
        novelty_level=1,
        problem_type="B",
        difficulty=difficulty,
        problem_statement=statement,
        num_steps=5,
        step_descriptions=[
            "Compute subtotal = price * quantity",
            "Apply discount to subtotal",
            "Apply tax after discount",
            "Add shipping to get total",
            "Compute per-unit cost",
        ],
        ground_truth_steps=[
            str(subtotal),
            str(after_discount),
            str(after_tax),
            str(total),
            str(per_unit),
        ],
        ground_truth_final=str(total),
        difficulty_metadata={
            "price_per_unit": p["price_per_unit"],
            "units": p["units"],
            "tax_pct": p["tax_pct"],
            "discount_pct": p["discount_pct"],
            "shipping": p["shipping"],
        },
    )


# ---------------------------------------------------------------------------
# L1 Type-C: Modular arithmetic chains (5 steps)
# ---------------------------------------------------------------------------


def generate_type_c_l1(
    problem_id: str,
    difficulty: str = "easy",
    seed: int = 42,
) -> Problem:
    """Generate an L1 Type-C problem: chain of modular arithmetic operations."""
    rng = random.Random(seed)

    mod_values = {"easy": 7, "medium": 13, "hard": 23}
    modulus = mod_values[difficulty]

    # Generate a chain of operations
    a = rng.randint(10, 99)
    b = rng.randint(10, 99)
    c = rng.randint(2, 15)
    d = rng.randint(10, 99)
    e = rng.randint(2, 8)

    # Step 1: a mod m
    s1 = a % modulus
    # Step 2: (s1 + b) mod m
    s2 = (s1 + b) % modulus
    # Step 3: (s2 * c) mod m
    s3 = (s2 * c) % modulus
    # Step 4: (s3 + d) mod m
    s4 = (s3 + d) % modulus
    # Step 5: (s4 ** e) mod m
    s5 = pow(s4, e, modulus)

    statement = (
        f"Perform the following chain of modular arithmetic operations (mod {modulus}).\n"
        f"Show your work for each step.\n\n"
        f"Step 1: Compute {a} mod {modulus}\n"
        f"Step 2: Add {b} to the result of Step 1, then take mod {modulus}\n"
        f"Step 3: Multiply the result of Step 2 by {c}, then take mod {modulus}\n"
        f"Step 4: Add {d} to the result of Step 3, then take mod {modulus}\n"
        f"Step 5: Raise the result of Step 4 to the power {e}, then take mod {modulus}"
    )

    return Problem(
        id=problem_id,
        novelty_level=1,
        problem_type="C",
        difficulty=difficulty,
        problem_statement=statement,
        num_steps=5,
        step_descriptions=[
            f"Compute {a} mod {modulus}",
            f"({s1} + {b}) mod {modulus}",
            f"({s2} * {c}) mod {modulus}",
            f"({s3} + {d}) mod {modulus}",
            f"({s4}^{e}) mod {modulus}",
        ],
        ground_truth_steps=[str(s1), str(s2), str(s3), str(s4), str(s5)],
        ground_truth_final=str(s5),
        difficulty_metadata={
            "modulus": modulus,
            "operands": [a, b, c, d, e],
        },
    )


# ---------------------------------------------------------------------------
# L2: Novel-operator variants
# ---------------------------------------------------------------------------

OPERATOR_PREAMBLE_ALPHA = (
    "CUSTOM OPERATOR DEFINITIONS (use these instead of standard arithmetic):\n"
    "  Zeta-addition:       a (+) b = a + b + 1\n"
    "  Zeta-subtraction:    a (-) b = a - b - 1\n"
    "  Zeta-multiplication: a (*) b = a * b + a + b\n"
    "  Standard division is unchanged.\n\n"
    "IMPORTANT: You MUST use these custom operators for EVERY arithmetic\n"
    "operation in your solution. Do NOT use standard +, -, or *.\n"
)

OPERATOR_PREAMBLE_BETA = (
    "CUSTOM OPERATOR DEFINITIONS (use these instead of standard arithmetic):\n"
    "  Star-addition:       a (+) b = 2*a + b  (NOTE: NOT commutative)\n"
    "  Star-subtraction:    a (-) b = 2*a - b\n"
    "  Standard multiplication and division are unchanged.\n\n"
    "IMPORTANT: You MUST use Star-addition and Star-subtraction\n"
    "for ALL addition/subtraction operations. Order matters!\n"
)


def _zeta_add(a, b):
    return a + b + 1


def _zeta_sub(a, b):
    return a - b - 1


def _zeta_mul(a, b):
    return a * b + a + b


def _star_add(a, b):
    return 2 * a + b


def _star_sub(a, b):
    return 2 * a - b


def generate_type_b_l2(
    problem_id: str,
    difficulty: str = "easy",
    seed: int = 42,
) -> Problem:
    """Generate an L2 Type-B problem: word problem with Zeta-operators."""
    rng = random.Random(seed)

    templates = {
        "easy": {
            "price_per_unit": rng.randint(3, 12),
            "units": rng.randint(5, 15),
            "tax_pct": rng.choice([5, 8, 10]),
            "discount_flat": rng.randint(5, 15),
            "shipping": rng.randint(3, 10),
        },
        "medium": {
            "price_per_unit": rng.randint(15, 50),
            "units": rng.randint(10, 30),
            "tax_pct": rng.choice([7, 9, 12, 15]),
            "discount_flat": rng.randint(10, 40),
            "shipping": rng.randint(8, 25),
        },
        "hard": {
            "price_per_unit": rng.randint(25, 100),
            "units": rng.randint(15, 50),
            "tax_pct": rng.choice([6, 8, 11, 14]),
            "discount_flat": rng.randint(20, 80),
            "shipping": rng.randint(12, 45),
        },
    }
    p = templates[difficulty]

    # Using Zeta operators for all arithmetic
    # Step 1: subtotal = price (*) quantity  (zeta-mul)
    subtotal = _zeta_mul(p["price_per_unit"], p["units"])
    # Step 2: after_discount = subtotal (-) discount  (zeta-sub)
    after_discount = _zeta_sub(subtotal, p["discount_flat"])
    # Step 3: tax_amount = after_discount (*) tax_rate_decimal ... simplified
    #         we compute tax as integer: floor(after_discount * tax_pct / 100)
    #         then after_tax = after_discount (+) tax_amount  (zeta-add)
    tax_amount = (after_discount * p["tax_pct"]) // 100
    after_tax = _zeta_add(after_discount, tax_amount)
    # Step 4: total = after_tax (+) shipping  (zeta-add)
    total = _zeta_add(after_tax, p["shipping"])
    # Step 5: Per-unit cost using standard division (total / units), rounded
    per_unit = round(total / p["units"], 2)

    statement = (
        f"{OPERATOR_PREAMBLE_ALPHA}"
        f"A customer orders {p['units']} items at ${p['price_per_unit']} each.\n"
        f"Use Zeta-multiplication to compute the subtotal (price (*) quantity).\n"
        f"Then apply a flat discount of ${p['discount_flat']} using Zeta-subtraction.\n"
        f"Compute the tax amount as floor(after_discount * {p['tax_pct']} / 100) using standard arithmetic,\n"
        f"then add it using Zeta-addition.\n"
        f"Add the shipping fee of ${p['shipping']} using Zeta-addition.\n"
        f"Finally, compute the per-unit cost (total / quantity, standard division).\n\n"
        "Calculate each step, showing your work:\n"
        "Step 1: Compute subtotal using Zeta-multiplication\n"
        "Step 2: Apply discount using Zeta-subtraction\n"
        "Step 3: Compute tax and apply using Zeta-addition\n"
        "Step 4: Add shipping using Zeta-addition\n"
        "Step 5: Compute per-unit cost (total / quantity)"
    )

    return Problem(
        id=problem_id,
        novelty_level=2,
        problem_type="B",
        difficulty=difficulty,
        problem_statement=statement,
        num_steps=5,
        step_descriptions=[
            f"Zeta-multiply: {p['price_per_unit']} (*) {p['units']}",
            f"Zeta-subtract discount: {subtotal} (-) {p['discount_flat']}",
            f"Compute tax and Zeta-add: {after_discount} (+) {tax_amount}",
            f"Zeta-add shipping: {after_tax} (+) {p['shipping']}",
            "Per-unit cost (total / quantity)",
        ],
        ground_truth_steps=[
            str(subtotal),
            str(after_discount),
            str(after_tax),
            str(total),
            str(per_unit),
        ],
        ground_truth_final=str(total),
        difficulty_metadata={
            "price_per_unit": p["price_per_unit"],
            "units": p["units"],
            "tax_pct": p["tax_pct"],
            "discount_flat": p["discount_flat"],
            "shipping": p["shipping"],
            "operator_set": "alpha",
        },
        operator_preamble=OPERATOR_PREAMBLE_ALPHA,
    )


def generate_type_c_l2(
    problem_id: str,
    difficulty: str = "easy",
    seed: int = 42,
) -> Problem:
    """Generate an L2 Type-C problem: modular arithmetic with Star-operators."""
    rng = random.Random(seed)

    mod_values = {"easy": 7, "medium": 13, "hard": 23}
    modulus = mod_values[difficulty]

    a = rng.randint(10, 99)
    b = rng.randint(10, 99)
    c = rng.randint(2, 15)
    d = rng.randint(10, 99)
    e = rng.randint(2, 8)

    # Step 1: a mod m  (standard)
    s1 = a % modulus
    # Step 2: Star-add(s1, b) mod m
    s2 = _star_add(s1, b) % modulus
    # Step 3: (s2 * c) mod m  (standard multiplication)
    s3 = (s2 * c) % modulus
    # Step 4: Star-add(s3, d) mod m
    s4 = _star_add(s3, d) % modulus
    # Step 5: (s4 ** e) mod m  (standard exponentiation)
    s5 = pow(s4, e, modulus)

    statement = (
        f"{OPERATOR_PREAMBLE_BETA}"
        f"Perform the following chain of operations (all results taken mod {modulus}).\n"
        f"Show your work for each step.\n\n"
        f"Step 1: Compute {a} mod {modulus}\n"
        f"Step 2: Apply Star-addition: result_of_step_1 (+) {b}, then take mod {modulus}\n"
        f"  (Remember: a (+) b = 2*a + b, NOT a + b)\n"
        f"Step 3: Multiply the result of Step 2 by {c} (standard multiplication), then take mod {modulus}\n"
        f"Step 4: Apply Star-addition: result_of_step_3 (+) {d}, then take mod {modulus}\n"
        f"Step 5: Raise the result of Step 4 to the power {e} (standard), then take mod {modulus}"
    )

    return Problem(
        id=problem_id,
        novelty_level=2,
        problem_type="C",
        difficulty=difficulty,
        problem_statement=statement,
        num_steps=5,
        step_descriptions=[
            f"Compute {a} mod {modulus}",
            f"Star-add({s1}, {b}) mod {modulus} = (2*{s1} + {b}) mod {modulus}",
            f"({s2} * {c}) mod {modulus}",
            f"Star-add({s3}, {d}) mod {modulus} = (2*{s3} + {d}) mod {modulus}",
            f"({s4}^{e}) mod {modulus}",
        ],
        ground_truth_steps=[str(s1), str(s2), str(s3), str(s4), str(s5)],
        ground_truth_final=str(s5),
        difficulty_metadata={
            "modulus": modulus,
            "operands": [a, b, c, d, e],
            "operator_set": "beta",
        },
        operator_preamble=OPERATOR_PREAMBLE_BETA,
    )


def generate_type_a_l2(
    problem_id: str,
    difficulty: str = "easy",
    seed: int = 42,
) -> Problem:
    """Generate an L2 Type-A problem: system of equations with Zeta-operators.

    Structurally mirrors L1 Type-A (solving for unknowns via elimination)
    so that the novelty comparison is valid. The model must solve a 3x3
    system where Zeta-multiplication replaces standard multiplication in
    the coefficient-variable products.

    Zeta-mul(a, x) = a*x + a + x, so each equation becomes:
      (a1*x + a1 + x) + (a2*y + a2 + y) + (a3*z + a3 + z) = rhs
    which simplifies to:
      (a1+1)*x + (a2+1)*y + (a3+1)*z = rhs - (a1 + a2 + a3)

    We generate the system in terms of the "effective" coefficients and
    present it to the model using Zeta-mul notation, requiring 5 elimination
    steps that parallel the L1 Type-A structure.
    """
    rng = random.Random(seed)

    ranges = {
        "easy": ((-3, 3), (-5, 5)),
        "medium": ((-5, 7), (-10, 10)),
        "hard": ((-8, 8), (-15, 15)),
    }
    coeff_range, sol_range = ranges[difficulty]

    # Generate "Zeta coefficients" a_ij for the system
    # The actual equation is: zeta_mul(a_i1, x) + zeta_mul(a_i2, y) + zeta_mul(a_i3, z) = rhs_i
    # Which expands to: (a_i1+1)*x + (a_i2+1)*y + (a_i3+1)*z + (a_i1+a_i2+a_i3) = rhs_i
    # So the "effective" standard system is A_eff * [x,y,z]^T = b_eff
    # where A_eff[i][j] = a_ij + 1, b_eff[i] = rhs_i - sum(a_ij)

    # We work backwards: pick solution and Zeta coefficients, compute RHS
    while True:
        A_zeta = [[rng.randint(*coeff_range) for _ in range(3)] for _ in range(3)]
        # Effective coefficients: a_ij + 1
        A_eff = [[A_zeta[i][j] + 1 for j in range(3)] for i in range(3)]
        # Ensure A_eff[0][0] != 0 for elimination
        if A_eff[0][0] == 0:
            continue
        det = (
            A_eff[0][0] * (A_eff[1][1] * A_eff[2][2] - A_eff[1][2] * A_eff[2][1])
            - A_eff[0][1] * (A_eff[1][0] * A_eff[2][2] - A_eff[1][2] * A_eff[2][0])
            + A_eff[0][2] * (A_eff[1][0] * A_eff[2][1] - A_eff[1][1] * A_eff[2][0])
        )
        if det != 0:
            break

    x_true = [rng.randint(*sol_range) for _ in range(3)]
    # Compute RHS for each equation:
    # rhs_i = sum(zeta_mul(a_ij, x_j)) for j=0..2
    rhs = []
    for i in range(3):
        row_sum = sum(_zeta_mul(A_zeta[i][j], x_true[j]) for j in range(3))
        rhs.append(row_sum)

    # Build the 5-step elimination using effective coefficients (same as L1)
    F = Fraction
    a00, a01, a02 = [F(c) for c in A_eff[0]]
    a10, a11, a12 = [F(c) for c in A_eff[1]]
    a20, a21, a22 = [F(c) for c in A_eff[2]]
    # Effective RHS: rhs_i - sum(A_zeta[i])
    b_eff = [rhs[i] - sum(A_zeta[i]) for i in range(3)]
    b0, b1, b2 = F(b_eff[0]), F(b_eff[1]), F(b_eff[2])

    # Step 1: Multiplier m21 = A_eff[1][0] / A_eff[0][0]
    m21 = a10 / a00
    m21_f = float(m21)
    gt_1 = f"{m21_f:.6g}"

    # Step 2: New RHS of eq2 after eliminating x
    new_b1 = b1 - m21 * b0
    gt_2 = f"{float(new_b1):.6g}"

    # Step 3: Eliminate x from eq3
    m31 = a20 / a00
    new_b2 = b2 - m31 * b0
    gt_3 = f"{float(new_b2):.6g}"

    # Step 4: Solve for z
    gt_4 = str(x_true[2])

    # Step 5: Back-substitute for y
    gt_5 = str(x_true[1])

    # Format Zeta equations for the problem statement
    def _fmt_zeta_eq(coeffs, rhs_val):
        parts = []
        var_names = ["x", "y", "z"]
        for j, c in enumerate(coeffs):
            term = f"{c} (*) {var_names[j]}"
            if parts:
                parts.append(f" + {term}")
            else:
                parts.append(term)
        return "".join(parts) + f" = {rhs_val}"

    eq_strs = [_fmt_zeta_eq(A_zeta[i], rhs[i]) for i in range(3)]

    statement = (
        f"{OPERATOR_PREAMBLE_ALPHA}"
        "Solve the following system of Zeta-equations for x, y, and z.\n"
        "Each term uses Zeta-multiplication: a (*) v = a*v + a + v.\n"
        "To solve, first expand each Zeta-product to get a standard linear system,\n"
        "then apply Gaussian elimination.\n\n"
        f"Equation 1: {eq_strs[0]}\n"
        f"Equation 2: {eq_strs[1]}\n"
        f"Equation 3: {eq_strs[2]}\n\n"
        "Steps (after expanding to standard form):\n"
        f"Step 1: Compute the multiplier m = eff_A[1][0]/eff_A[0][0] = {A_eff[1][0]}/{A_eff[0][0]}\n"
        "Step 2: Eliminate x from equation 2 using the multiplier. State the new RHS.\n"
        f"Step 3: Eliminate x from equation 3 (multiplier = {A_eff[2][0]}/{A_eff[0][0]}). State the new RHS.\n"
        "Step 4: Solve the reduced 2x2 system for z.\n"
        "Step 5: Back-substitute z to find y."
    )

    return Problem(
        id=problem_id,
        novelty_level=2,
        problem_type="A",
        difficulty=difficulty,
        problem_statement=statement,
        num_steps=5,
        step_descriptions=[
            f"Compute multiplier m = {A_eff[1][0]}/{A_eff[0][0]}",
            f"Eliminate x from eq2: new RHS = {float(b1):.6g} - {m21_f:.6g}*{float(b0):.6g}",
            f"Eliminate x from eq3: new RHS = {float(b2):.6g} - {float(m31):.6g}*{float(b0):.6g}",
            "Solve reduced 2x2 system for z",
            "Back-substitute z to find y",
        ],
        ground_truth_steps=[gt_1, gt_2, gt_3, gt_4, gt_5],
        ground_truth_final=f"x = {x_true[0]}, y = {x_true[1]}, z = {x_true[2]}",
        difficulty_metadata={
            "zeta_coefficients": A_zeta,
            "effective_coefficients": A_eff,
            "solution": x_true,
            "rhs": rhs,
            "determinant": det,
            "operator_set": "alpha",
        },
        operator_preamble=OPERATOR_PREAMBLE_ALPHA,
    )


# ---------------------------------------------------------------------------
# Problem set generation
# ---------------------------------------------------------------------------


def generate_problem_set(
    novelty_level: int = 1,
    base_seed: int = 42,
    num_main: int = 10,
    num_feedback: int = 5,
) -> list[Problem]:
    """Generate a complete problem set for one novelty level.

    Produces *num_main* problems for the main tasks (Tasks 1-4, 6) and
    *num_feedback* additional problems for the feedback/adaptive task (Task 5).
    Problems are distributed across types A, B, C and difficulties easy/medium/hard.
    """
    problems: list[Problem] = []
    gen_l1 = {
        "A": generate_type_a_l1,
        "B": generate_type_b_l1,
        "C": generate_type_c_l1,
    }
    gen_l2 = {
        "A": generate_type_a_l2,
        "B": generate_type_b_l2,
        "C": generate_type_c_l2,
    }
    generators = gen_l1 if novelty_level == 1 else gen_l2
    prefix = "l1" if novelty_level == 1 else "l2"
    difficulties = ["easy", "medium", "hard"]
    types = ["A", "B", "C"]

    idx = 0
    # Main problems
    # Use i//3 for difficulty so each "round" of 3 types shares a difficulty,
    # and successive rounds advance the difficulty. This breaks the coupling
    # that previously locked Type A = easy, Type B = medium, Type C = hard.
    for i in range(num_main):
        ptype = types[i % len(types)]
        diff = difficulties[(i // len(types)) % len(difficulties)]
        pid = f"{prefix}_main_{idx:03d}"
        problems.append(generators[ptype](pid, diff, base_seed + idx))
        idx += 1

    # Feedback problems — same decoupling
    for i in range(num_feedback):
        ptype = types[i % len(types)]
        diff = difficulties[(i // len(types)) % len(difficulties)]
        pid = f"{prefix}_feedback_{i:03d}"
        problems.append(generators[ptype](pid, diff, base_seed + 1000 + i))

    return problems


def generate_all_problems(
    base_seed: int = 42,
    num_main: int = 10,
    num_feedback: int = 5,
) -> dict:
    """Generate and return both L1 and L2 problem sets as serializable dicts."""
    l1 = generate_problem_set(1, base_seed, num_main, num_feedback)
    l2 = generate_problem_set(2, base_seed + 500, num_main, num_feedback)
    return {
        "l1": [p.to_dict() for p in l1],
        "l2": [p.to_dict() for p in l2],
    }


def save_problems(output_dir: str = ".", base_seed: int = 42):
    """Generate all problems and save to JSON files."""
    import os

    os.makedirs(output_dir, exist_ok=True)
    data = generate_all_problems(base_seed)
    l1_path = os.path.join(output_dir, "l1_problems.json")
    l2_path = os.path.join(output_dir, "l2_problems.json")
    with open(l1_path, "w") as f:
        json.dump(data["l1"], f, indent=2)
    with open(l2_path, "w") as f:
        json.dump(data["l2"], f, indent=2)
    return l1_path, l2_path


if __name__ == "__main__":
    save_problems(".")
    print("Problems generated successfully.")
