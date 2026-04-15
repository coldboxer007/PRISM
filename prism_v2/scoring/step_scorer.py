"""
Step Scorer: Extract and verify per-step answers from LLM solving responses.

Handles:
  - Extracting intermediate answers from step-by-step solutions
  - Comparing extracted answers against ground truth
  - Numeric comparison with tolerance for floating point
"""

import re
from typing import Optional


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_step_answers(response_text: str, num_steps: int) -> list[str]:
    """Extract intermediate answers for each step from the model's solution.

    Looks for patterns like:
      - "Step N: ... = ANSWER"
      - "Step N: ... answer is ANSWER"
      - "Step N: ... result: ANSWER"
      - Boxed answers: \\boxed{ANSWER}

    Returns a list of extracted answer strings, one per step.
    Missing steps get an empty string.
    """
    step_answers = [""] * num_steps

    # Strategy 1: Split by "Step N" headers and extract the last number/expression
    step_blocks = _split_into_step_blocks(response_text, num_steps)

    for i, block in enumerate(step_blocks):
        if i >= num_steps:
            break
        answer = _extract_answer_from_block(block)
        if answer:
            step_answers[i] = answer

    return step_answers


def _split_into_step_blocks(text: str, num_steps: int) -> list[str]:
    """Split response text into blocks, one per step.

    Uses the actual step number from 'Step N' headers for alignment,
    so skipped or out-of-order steps map to the correct index.
    Duplicate step numbers keep the later occurrence (self-correction).
    """
    # Find all "Step N" positions
    pattern = r"(?:^|\n)\s*(?:\*\*)?Step\s+(\d+)\s*(?:\*\*)?[:\.]"
    matches = list(re.finditer(pattern, text, re.IGNORECASE))

    if not matches:
        return [""] * num_steps

    blocks = [""] * num_steps
    for i, m in enumerate(matches):
        step_num = int(m.group(1))
        if step_num < 1 or step_num > num_steps:
            continue  # skip out-of-range step numbers

        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # Check for FINAL ANSWER section and truncate
        final_marker = re.search(
            r"(?:FINAL\s+ANSWER|Final\s+Answer)",
            text[start:end],
            re.IGNORECASE,
        )
        if final_marker:
            end = start + final_marker.start()

        # Use step_num for indexing; later duplicates overwrite earlier ones
        blocks[step_num - 1] = text[start:end]

    return blocks


def _extract_answer_from_block(block: str) -> str:
    """Extract the most likely answer from a step block.

    Priority:
      1. Boxed answer: \\boxed{...}
      2. Explicit marker: "= ANSWER", "answer is ANSWER", "result: ANSWER"
      3. Last number in the block
    """
    block = block.strip()
    if not block:
        return ""

    # Priority 1: Boxed answer
    boxed = re.findall(r"\\boxed\{([^}]+)\}", block)
    if boxed:
        return boxed[-1].strip()

    # Priority 2: Explicit "= answer" at end of a line
    eq_matches = re.findall(
        r"=\s*([-\d.,/\s]+(?:\.\d+)?)\s*$",
        block,
        re.MULTILINE,
    )
    if eq_matches:
        return eq_matches[-1].strip().rstrip(".")

    # Priority 2b: "answer is X" or "result is X" or "result: X"
    explicit = re.findall(
        r"(?:answer|result|gives|equals)\s*(?:is|:)?\s*([-\d.,/\s]+(?:\.\d+)?)",
        block,
        re.IGNORECASE,
    )
    if explicit:
        return explicit[-1].strip().rstrip(".")

    # Priority 3: Last number/expression on any line
    numbers = re.findall(r"([-]?\d+(?:[.,/]\d+)*)", block)
    if numbers:
        return numbers[-1].strip()

    return ""


def extract_final_answer(response_text: str) -> str:
    """Extract the final answer from the model's solution response."""
    # Look for explicit FINAL ANSWER marker
    pattern = r"(?:FINAL\s+ANSWER|Final\s+Answer)\s*[:\s]*(.+?)(?:\n\n|\Z)"
    m = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
    if m:
        answer = m.group(1).strip()
        # Clean up: remove markdown bold markers
        answer = re.sub(r"\*\*", "", answer)
        return answer

    # Explicit UNSOLVABLE / no-solution declaration
    unsolvable = re.search(
        r"\b(?:UNSOLVABLE|NO\s+SOLUTION|INCONSISTENT|CANNOT\s+BE\s+SOLVED)\b",
        response_text,
        re.IGNORECASE,
    )
    if unsolvable:
        return "UNSOLVABLE"

    # Fallback: look for boxed answer anywhere
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response_text)
    if boxed:
        return boxed[-1].strip()

    # Fallback: last line with "=" or a number
    lines = response_text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if "=" in line and any(c.isdigit() for c in line):
            parts = line.split("=")
            return parts[-1].strip()

    return ""


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------


def _normalize_number(s: str) -> Optional[float]:
    """Try to parse a string as a number. Returns None if not parseable."""
    s = s.strip().replace(",", "").replace(" ", "")
    # Handle fractions like "3/4"
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2:
            try:
                return float(parts[0]) / float(parts[1])
            except (ValueError, ZeroDivisionError):
                return None
    try:
        return float(s)
    except ValueError:
        return None


def _extract_variable_assignments(s: str) -> dict[str, float]:
    """Extract assignments like ``x = 1, y = 2, z = 3``."""
    matches = re.findall(r"\b([xyz])\s*=\s*([-\d.,/]+)", s, re.IGNORECASE)
    assignments: dict[str, float] = {}
    for var, raw_value in matches:
        value = _normalize_number(raw_value)
        if value is not None:
            assignments[var.lower()] = value
    return assignments


def _extract_number_sequence(s: str) -> list[float]:
    """Extract a loose ordered number sequence from a string."""
    values = []
    for raw_value in re.findall(r"[-]?\d+(?:[.,/]\d+)*", s):
        value = _normalize_number(raw_value)
        if value is not None:
            values.append(value)
    return values


def compare_answers(
    extracted: str,
    ground_truth: str,
    tolerance: float = 0.01,
) -> bool:
    """Compare an extracted answer against ground truth.

    Tries numeric comparison first (with tolerance), then falls back
    to normalized string comparison.
    """
    if not extracted or not ground_truth:
        return False

    # Clean whitespace
    extracted_clean = extracted.strip().lower()
    truth_clean = ground_truth.strip().lower()

    # Direct string match
    if extracted_clean == truth_clean:
        return True

    # Structured variable assignments, e.g. x=1, y=2, z=3
    extracted_assignments = _extract_variable_assignments(extracted_clean)
    truth_assignments = _extract_variable_assignments(truth_clean)
    if truth_assignments:
        if extracted_assignments:
            truth_keys = sorted(truth_assignments.keys())
            if sorted(extracted_assignments.keys()) == truth_keys:
                return all(
                    compare_answers(
                        str(extracted_assignments[key]),
                        str(truth_assignments[key]),
                        tolerance,
                    )
                    for key in truth_keys
                )

        # Accept tuple-style final answers if the truth specifies x, y, z.
        if sorted(truth_assignments.keys()) == ["x", "y", "z"]:
            extracted_sequence = _extract_number_sequence(extracted_clean)
            if len(extracted_sequence) >= 3:
                return all(
                    compare_answers(
                        str(extracted_sequence[i]),
                        str(truth_assignments[key]),
                        tolerance,
                    )
                    for i, key in enumerate(["x", "y", "z"])
                )

    # Try numeric comparison
    ext_num = _normalize_number(extracted)
    truth_num = _normalize_number(ground_truth)

    if ext_num is not None and truth_num is not None:
        if truth_num == 0:
            return abs(ext_num) < tolerance
        return abs(ext_num - truth_num) / max(abs(truth_num), 1e-10) < tolerance

    # Normalize: remove spaces, commas, dollar signs for comparison
    def normalize(s):
        return re.sub(r"[\s,$=]", "", s.lower())

    return normalize(extracted_clean) == normalize(truth_clean)


def score_steps(
    extracted_answers: list[str],
    ground_truth_answers: list[str],
    tolerance: float = 0.01,
) -> list[bool]:
    """Score each step answer against ground truth.

    Returns a list of booleans (True = correct, False = incorrect).
    """
    n = min(len(extracted_answers), len(ground_truth_answers))
    results = []
    for i in range(n):
        results.append(
            compare_answers(extracted_answers[i], ground_truth_answers[i], tolerance)
        )
    # Pad with False if extracted has fewer answers
    while len(results) < len(ground_truth_answers):
        results.append(False)
    return results
