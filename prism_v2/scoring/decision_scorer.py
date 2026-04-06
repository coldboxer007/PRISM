"""
Decision Response Parser for Metacognitive Control (Task 5).

Extracts the model's ACCEPT/DECLINE decision, stated confidence,
and reasoning from a decision-task response.
"""

import re
from dataclasses import dataclass, field


@dataclass
class DecisionResponse:
    """Parsed output from a metacognitive decision prompt."""

    decision: str = "accept"  # "accept" or "decline"
    confidence: int = 50  # 0-100
    reasoning: str = ""
    parse_errors: list[str] = field(default_factory=list)


def parse_decision_response(response_text: str) -> DecisionResponse:
    """Parse a decision-task response.

    Extracts:
      - DECISION: ACCEPT or DECLINE
      - CONFIDENCE: 0-100
      - REASONING: free text
    """
    result = DecisionResponse()

    # --- Extract decision ---
    dec_match = re.search(
        r"DECISION\s*:\s*(ACCEPT|DECLINE)",
        response_text,
        re.IGNORECASE,
    )
    if dec_match:
        result.decision = dec_match.group(1).lower()
    else:
        # Fallback: look for standalone ACCEPT or DECLINE
        if re.search(r"\bDECLINE\b", response_text, re.IGNORECASE):
            result.decision = "decline"
        elif re.search(r"\bACCEPT\b", response_text, re.IGNORECASE):
            result.decision = "accept"
        else:
            result.decision = "accept"  # default
            result.parse_errors.append(
                "Could not extract DECISION; defaulting to accept"
            )

    # --- Extract confidence ---
    conf_match = re.search(
        r"CONFIDENCE\s*:\s*(\d+)\s*%?",
        response_text,
        re.IGNORECASE,
    )
    if conf_match:
        result.confidence = min(100, max(0, int(conf_match.group(1))))
    else:
        # Fallback: look for percentage near "confident"
        conf_fallback = re.search(r"(\d{1,3})\s*%", response_text)
        if conf_fallback:
            result.confidence = min(100, max(0, int(conf_fallback.group(1))))
        else:
            result.confidence = 50
            result.parse_errors.append("Could not extract CONFIDENCE; defaulting to 50")

    # --- Extract reasoning ---
    reason_match = re.search(
        r"REASONING\s*:\s*(.+?)(?:\n\n|\n(?:Step|Then|Now|Attempt)|\Z)",
        response_text,
        re.IGNORECASE | re.DOTALL,
    )
    if reason_match:
        result.reasoning = reason_match.group(1).strip()
    else:
        result.parse_errors.append("Could not extract REASONING")

    return result
