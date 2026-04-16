"""
Confidence Parser: Extract structured metacognitive data from LLM responses.

Handles parsing of:
  - Dimension 1 (Prospective): weakest step, per-step confidence, bet fraction
  - Dimension 3a (Blind Retrospective): per-step self-assessment before results
  - Dimension 3b (Informed Retrospective): hardest step, counterfactual
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Confidence label mappings
# ---------------------------------------------------------------------------

CONFIDENCE_LABELS = [
    "definitely wrong",
    "probably wrong",
    "uncertain",
    "probably right",
    "definitely right",
]

CONFIDENCE_TO_INT = {label: i + 1 for i, label in enumerate(CONFIDENCE_LABELS)}
# {"definitely wrong": 1, "probably wrong": 2, "uncertain": 3,
#  "probably right": 4, "definitely right": 5}

RETRO_LABELS = [
    "confident and correct",
    "confident but wrong",
    "uncertain and correct",
    "uncertain and wrong",
]

RETRO_TO_DIFFICULTY = {
    "confident and correct": 1,
    "uncertain and correct": 2,
    "confident but wrong": 3,
    "uncertain and wrong": 4,
}


# ---------------------------------------------------------------------------
# Parsed data structures
# ---------------------------------------------------------------------------


@dataclass
class ProspectiveReport:
    """Parsed data from a Dimension 1 (prospective) response."""

    predicted_weakest_step: Optional[int] = None
    confidence_vector: list[int] = field(default_factory=list)
    confidence_labels: list[str] = field(default_factory=list)
    bet_fraction_correct: Optional[float] = None
    parse_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return (
            self.predicted_weakest_step is not None
            and len(self.confidence_vector) > 0
            and self.bet_fraction_correct is not None
        )


@dataclass
class BlindRetrospectiveReport:
    """Parsed data from a D3a (blind retrospective) response.

    Contains the model's per-step self-assessment made BEFORE
    seeing which steps were correct/incorrect.
    """

    self_assessment: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.self_assessment) > 0


@dataclass
class RetrospectiveReport:
    """Parsed data from a D3b (informed retrospective) response.

    Contains hardest step and counterfactual text, produced AFTER
    the model sees which steps were correct/incorrect.
    """

    reported_hardest_step: Optional[int] = None
    counterfactual_text: str = ""
    parse_errors: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.reported_hardest_step is not None


# ---------------------------------------------------------------------------
# Parsing functions
# ---------------------------------------------------------------------------


def _extract_weakest_step(
    text: str, section_header: str = "WEAKEST STEP"
) -> Optional[int]:
    """Extract the predicted/reported weakest/hardest step number."""
    # Pattern 1: After the header, find a standalone integer
    pattern = rf"{section_header}[:\s]*(\d+)"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        return int(m.group(1))

    # Pattern 2: "Step X" after the header
    pattern2 = rf"{section_header}.*?(?:step\s+)?(\d+)"
    m2 = re.search(pattern2, text, re.IGNORECASE | re.DOTALL)
    if m2:
        return int(m2.group(1))

    # Pattern 3: Numbered-list format "1. N" or "1: N" — GPT-5.4 style
    # The model answers the first prompt question with just the step number
    # under a "1." list marker instead of repeating the header text.
    pattern3 = r"(?:^|\n)\s*1[.:]\s*(?:step\s+)?(\d+)\s*(?:\n|$)"
    m3 = re.search(pattern3, text, re.IGNORECASE | re.MULTILINE)
    if m3:
        return int(m3.group(1))

    return None


def _extract_confidence_vector(
    text: str, num_steps: int
) -> tuple[list[int], list[str]]:
    """Extract per-step confidence labels and convert to integer vector."""
    labels_pattern = "|".join(re.escape(l) for l in CONFIDENCE_LABELS)
    pattern = rf"Step\s+(\d+)\s*:\s*[\"']?({labels_pattern})[\"']?"
    matches = re.findall(pattern, text, re.IGNORECASE)

    vector = []
    labels = []
    # Build a mapping from step number to label
    step_map = {}
    for step_num_str, label in matches:
        step_num = int(step_num_str)
        label_lower = label.lower()
        step_map[step_num] = label_lower

    for i in range(1, num_steps + 1):
        if i in step_map:
            lab = step_map[i]
            vector.append(CONFIDENCE_TO_INT.get(lab, 3))
            labels.append(lab)
        else:
            vector.append(3)  # default to "uncertain"
            labels.append("uncertain")

    return vector, labels


def _extract_bet_fraction(text: str) -> Optional[float]:
    """Extract the bet fraction (amount bet on CORRECT / 100)."""
    # Pattern 1: "$X on correct" or "$X on CORRECT"
    pattern1 = r"\$?(\d+(?:\.\d+)?)\s*(?:on\s+)?correct"
    m1 = re.search(pattern1, text, re.IGNORECASE)
    if m1:
        return float(m1.group(1)) / 100.0

    # Pattern 2: "CORRECT: $X" or "correct: X"
    pattern2 = r"correct\s*[:=]\s*\$?(\d+(?:\.\d+)?)"
    m2 = re.search(pattern2, text, re.IGNORECASE)
    if m2:
        return float(m2.group(1)) / 100.0

    # Pattern 3: "bet $X that" or "wager $X"
    pattern3 = r"(?:bet|wager)\s*\$?(\d+(?:\.\d+)?)\s*(?:on\s+)?(?:correct|right|that)"
    m3 = re.search(pattern3, text, re.IGNORECASE)
    if m3:
        return float(m3.group(1)) / 100.0

    # Pattern 4: Split "X/Y" where X is correct amount
    pattern4 = r"(\d+)\s*/\s*(\d+)"
    m4 = re.search(pattern4, text)
    if m4:
        x, y = int(m4.group(1)), int(m4.group(2))
        if x + y == 100:
            return x / 100.0

    return None


def _extract_retro_assessment(text: str, num_steps: int) -> list[str]:
    """Extract per-step retrospective self-assessment labels."""
    labels_pattern = "|".join(re.escape(l) for l in RETRO_LABELS)
    pattern = rf"Step\s+(\d+)\s*:\s*[\"']?({labels_pattern})[\"']?"
    matches = re.findall(pattern, text, re.IGNORECASE)

    step_map = {}
    for step_num_str, label in matches:
        step_num = int(step_num_str)
        step_map[step_num] = label.lower()

    result = []
    for i in range(1, num_steps + 1):
        result.append(step_map.get(i, "uncertain and wrong"))
    return result


def _extract_counterfactual(text: str) -> str:
    """Extract the counterfactual text from D3b response."""
    # Try to find text after "COUNTERFACTUAL:" header
    pattern = r"(?:2\.\s*)?COUNTERFACTUAL\s*:\s*(.*?)(?:\n\n|\Z)"
    m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()

    # Fallback: look for text after "2." numbered item (D3b has 2 questions)
    pattern2 = r"(?:^|\n)\s*2\.\s*(.*?)(?:\n\n|\Z)"
    m2 = re.search(pattern2, text, re.DOTALL)
    if m2:
        candidate = m2.group(1).strip()
        # Avoid capturing the blind-retrospective labels section
        # (some models put "Step N: ..." under item 2)
        if candidate and not re.match(r"Step\s+\d+\s*:", candidate, re.IGNORECASE):
            return candidate

    # Fallback: "3." numbered item (GPT-5.4 style: 1=hardest, 2=labels, 3=counterfactual)
    pattern3 = r"(?:^|\n)\s*3\.\s*(.*?)(?:\n\n|\Z)"
    m3 = re.search(pattern3, text, re.DOTALL)
    if m3:
        candidate = m3.group(1).strip()
        if candidate:
            return candidate

    return ""


# ---------------------------------------------------------------------------
# Public parsing API
# ---------------------------------------------------------------------------


def parse_prospective(response_text: str, num_steps: int) -> ProspectiveReport:
    """Parse a Dimension 1 (prospective) response into structured data."""
    report = ProspectiveReport()

    # Extract weakest step
    report.predicted_weakest_step = _extract_weakest_step(response_text, "WEAKEST STEP")
    if report.predicted_weakest_step is None:
        report.predicted_weakest_step = _extract_weakest_step(response_text, "HARDEST")
    if report.predicted_weakest_step is None:
        report.parse_errors.append("Could not extract predicted weakest step")

    # Extract confidence vector
    vec, labs = _extract_confidence_vector(response_text, num_steps)
    report.confidence_vector = vec
    report.confidence_labels = labs
    if not any(v != 3 for v in vec):
        report.parse_errors.append("All confidence values defaulted to 'uncertain'")

    # Extract bet fraction
    report.bet_fraction_correct = _extract_bet_fraction(response_text)
    if report.bet_fraction_correct is None:
        report.bet_fraction_correct = 0.5  # default
        report.parse_errors.append("Could not extract bet fraction, defaulting to 0.5")

    return report


def parse_blind_retrospective(
    response_text: str, num_steps: int
) -> BlindRetrospectiveReport:
    """Parse a D3a (blind retrospective) response.

    Extracts per-step self-assessment labels produced BEFORE the model
    was told which steps were correct/incorrect.
    """
    report = BlindRetrospectiveReport()
    report.self_assessment = _extract_retro_assessment(response_text, num_steps)

    # Check if all defaulted (no labels found)
    if all(a == "uncertain and wrong" for a in report.self_assessment):
        report.parse_errors.append("No retrospective labels found in blind assessment")

    return report


def parse_retrospective(response_text: str, num_steps: int) -> RetrospectiveReport:
    """Parse a D3b (informed retrospective) response.

    Extracts hardest step and counterfactual text produced AFTER the
    model was told which steps were correct/incorrect.
    """
    report = RetrospectiveReport()

    # Extract hardest step
    report.reported_hardest_step = _extract_weakest_step(response_text, "HARDEST STEP")
    if report.reported_hardest_step is None:
        report.reported_hardest_step = _extract_weakest_step(response_text, "HARDEST")
    if report.reported_hardest_step is None:
        report.reported_hardest_step = _extract_weakest_step(
            response_text, "most trouble"
        )
    if report.reported_hardest_step is None:
        report.parse_errors.append("Could not extract reported hardest step")

    # Extract counterfactual
    report.counterfactual_text = _extract_counterfactual(response_text)
    if not report.counterfactual_text:
        report.parse_errors.append("Could not extract counterfactual text")

    return report
