"""
Step Scorer: Extract and verify per-step answers from LLM solving responses.

Handles:
  - Extracting intermediate answers from step-by-step solutions
  - Comparing extracted answers against ground truth
  - Numeric comparison with tolerance for floating point
"""

import re
from typing import Optional


_NUMBER_PATTERN = re.compile(
    r"[-+]?\d+(?:\.\d+)?(?:/\s*[-+]?\d+(?:\.\d+)?)?"
)
_UNSOLVABLE_PATTERN = re.compile(
    r"\b(?:UNSOLVABLE|NO\s+SOLUTION|INCONSISTENT|CANNOT\s+BE\s+SOLVED)\b",
    re.IGNORECASE,
)


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
    boxed = _extract_latex_macro_contents(block, "boxed")
    for content in reversed(boxed):
        candidate = _extract_numeric_candidate(content)
        if candidate:
            return candidate

    # Priority 2: explicit answer/result lines near the end of the block
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    for line in reversed(lines):
        lowered = line.lower()
        if (
            "=" in line
            or any(
                token in lowered
                for token in (
                    "answer",
                    "result",
                    "rhs",
                    "subtotal",
                    "discount",
                    "price after",
                    "tax amount",
                    "total including",
                    "cost per unit",
                    "multiplier",
                )
            )
        ):
            candidate = _extract_numeric_candidate(line)
            if candidate:
                return candidate

    # Priority 3: Last number/expression on any line
    candidate = _extract_numeric_candidate(block)
    if candidate:
        return candidate

    return ""


def extract_final_answer(response_text: str) -> str:
    """Extract the final answer from the model's solution response."""
    unsolvable = _extract_unsolvable_marker(response_text)
    if unsolvable:
        return unsolvable

    # Look for explicit FINAL ANSWER marker
    pattern = r"(?:FINAL\s+ANSWER|Final\s+Answer)\s*[:\s]*(.+?)(?:\n\n|\Z)"
    m = re.search(pattern, response_text, re.IGNORECASE | re.DOTALL)
    if m:
        answer = _extract_structured_final_answer(m.group(1))
        if answer:
            return answer

    # Fallback: look for boxed answer anywhere
    boxed = _extract_latex_macro_contents(response_text, "boxed")
    for content in reversed(boxed):
        candidate = _extract_numeric_candidate(content)
        if candidate:
            return candidate

    assignments = _extract_variable_assignments(response_text)
    if assignments:
        return _format_assignments(assignments)

    # Fallback: last line with "=" or a number
    lines = response_text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        candidate = _extract_numeric_candidate(line)
        if candidate:
            return candidate

    return ""


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------


def _normalize_number(s: str) -> Optional[float]:
    """Try to parse a string as a number. Returns None if not parseable."""
    s = _strip_common_wrappers(s)
    s = s.strip().replace(",", "").replace(" ", "")
    if "=" in s:
        rhs = s.split("=")[-1].strip()
        if _NUMBER_PATTERN.fullmatch(rhs):
            s = rhs

    if _NUMBER_PATTERN.fullmatch(s) is None:
        matches = _NUMBER_PATTERN.findall(s)
        if len(matches) == 1:
            s = matches[0].replace(" ", "")
        else:
            return None

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
    cleaned = _strip_common_wrappers(s)
    matches = re.findall(r"\b([xyz])\s*=\s*([-\d.,/]+)", cleaned, re.IGNORECASE)
    assignments: dict[str, float] = {}
    for var, raw_value in matches:
        value = _normalize_number(raw_value)
        if value is not None:
            assignments[var.lower()] = value
    return assignments


def _extract_number_sequence(s: str) -> list[float]:
    """Extract a loose ordered number sequence from a string."""
    values = []
    for raw_value in _NUMBER_PATTERN.findall(_strip_common_wrappers(s)):
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

    # Clean whitespace / common formatting wrappers
    extracted_clean = _strip_common_wrappers(extracted).strip().lower()
    truth_clean = _strip_common_wrappers(ground_truth).strip().lower()

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
        return re.sub(r"[\s,$={}]", "", _strip_common_wrappers(s).lower())

    return normalize(extracted_clean) == normalize(truth_clean)


def _extract_unsolvable_marker(text: str) -> str:
    """Return canonical UNSOLVABLE marker if the text declares no solution."""
    return "UNSOLVABLE" if _UNSOLVABLE_PATTERN.search(text or "") else ""


def _extract_latex_macro_contents(text: str, macro: str) -> list[str]:
    """Extract ``\\macro{...}`` contents with balanced-brace parsing."""
    contents = []
    needle = f"\\{macro}" + "{"
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            break
        cursor = idx + len(needle)
        depth = 1
        chunk = []
        while cursor < len(text) and depth > 0:
            char = text[cursor]
            if char == "{":
                depth += 1
                chunk.append(char)
            elif char == "}":
                depth -= 1
                if depth > 0:
                    chunk.append(char)
            else:
                chunk.append(char)
            cursor += 1
        if depth == 0:
            contents.append("".join(chunk))
            start = cursor
        else:
            break
    return contents


def _strip_common_wrappers(text: str) -> str:
    """Remove common markdown/LaTeX wrappers while preserving answer content."""
    if not text:
        return ""

    cleaned = str(text)
    cleaned = cleaned.replace("\u2212", "-").replace("−", "-").replace("–", "-")
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"\*\*", "", cleaned)
    cleaned = re.sub(
        r"\\frac\{([^{}]+)\}\{([^{}]+)\}",
        lambda m: f"{m.group(1)}/{m.group(2)}",
        cleaned,
    )

    for macro in ("text", "mathrm", "operatorname", "mathbf", "mathit", "mbox"):
        pattern = re.compile(rf"\\{macro}\{{([^{{}}]*)\}}")
        while True:
            updated = pattern.sub(r"\1", cleaned)
            if updated == cleaned:
                break
            cleaned = updated

    cleaned = re.sub(r"\\pmod\{([^{}]+)\}", r" mod \1", cleaned)
    cleaned = re.sub(r"\\(?:left|right|displaystyle|quad|qquad|,|!|;)", " ", cleaned)
    cleaned = cleaned.replace("\\(", " ").replace("\\)", " ")
    cleaned = cleaned.replace("\\[", " ").replace("\\]", " ")
    cleaned = re.sub(r"\\[a-zA-Z]+", " ", cleaned)
    cleaned = cleaned.replace("$", " ")
    cleaned = cleaned.replace("{", " ").replace("}", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _extract_numeric_candidate(text: str) -> str:
    """Extract the most likely numeric answer from a line or boxed fragment."""
    if not text:
        return ""

    if _extract_unsolvable_marker(text):
        return "UNSOLVABLE"

    cleaned = _strip_common_wrappers(text)
    if not cleaned:
        return ""

    assignments = _extract_variable_assignments(cleaned)
    if len(assignments) >= 2:
        return _format_assignments(assignments)

    if "=" in cleaned:
        for piece in reversed(cleaned.split("=")[1:]):
            matches = _NUMBER_PATTERN.findall(piece)
            if matches:
                return matches[-1].replace(" ", "")

    matches = _NUMBER_PATTERN.findall(cleaned)
    if matches:
        return matches[-1].replace(" ", "")

    return ""


def _extract_structured_final_answer(text: str) -> str:
    """Extract a final answer from the dedicated final-answer section."""
    if not text:
        return ""

    if _extract_unsolvable_marker(text):
        return "UNSOLVABLE"

    assignments = _extract_variable_assignments(text)
    if assignments:
        return _format_assignments(assignments)

    boxed = _extract_latex_macro_contents(text, "boxed")
    for content in reversed(boxed):
        candidate = _extract_numeric_candidate(content)
        if candidate:
            return candidate

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        candidate = _extract_numeric_candidate(line)
        if candidate:
            return candidate

    return _strip_common_wrappers(text)


def _format_assignments(assignments: dict[str, float]) -> str:
    """Format variable assignments deterministically for downstream comparison."""
    ordered = []
    for key in ("x", "y", "z"):
        if key in assignments:
            value = assignments[key]
            if float(value).is_integer():
                value_str = str(int(value))
            else:
                value_str = str(value)
            ordered.append(f"{key} = {value_str}")
    return ", ".join(ordered)


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
