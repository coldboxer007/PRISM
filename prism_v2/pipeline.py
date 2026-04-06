"""
PRISM v2.1 Pipeline Orchestrator

Runs the full D1→D2→D3 metacognitive assessment pipeline for each problem.
Implements result caching so multiple tasks can share a single pipeline run.

Usage in Kaggle notebook:
    pipeline = PrismPipeline(problems_l1, problems_l2)
    pipeline.run_all(llm)         # runs once and caches
    score = pipeline.get_task_1_score()  # reads from cache
"""

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Result structures
# ---------------------------------------------------------------------------


@dataclass
class ProblemResult:
    """Complete result record for a single problem across all dimensions."""

    problem_id: str
    novelty_level: int
    problem_type: str
    difficulty: str

    # Dimension 1: Prospective
    d1_predicted_weakest: Optional[int] = None
    d1_confidence_vector: list[int] = field(default_factory=list)
    d1_confidence_labels: list[str] = field(default_factory=list)
    d1_bet_correct: float = 0.5

    # Dimension 2: Performance
    d2_step_answers: list[str] = field(default_factory=list)
    d2_step_correct: list[bool] = field(default_factory=list)
    d2_overall_correct: bool = False
    d2_actual_weakest: Optional[int] = None
    d2_final_answer: str = ""

    # Dimension 3: Retrospective
    d3_reported_hardest: Optional[int] = None
    d3_self_assessment: list[str] = field(default_factory=list)
    d3_counterfactual: str = ""

    # Metadata
    parse_errors: list[str] = field(default_factory=list)
    num_steps: int = 5


@dataclass
class FeedbackRoundResult:
    """Result for a single feedback round."""

    problem_id: str
    round_number: int
    d1_confidence_vector: list[int] = field(default_factory=list)
    d1_confidence_labels: list[str] = field(default_factory=list)
    d1_bet_correct: float = 0.5
    d2_step_correct: list[bool] = field(default_factory=list)
    d2_overall_correct: bool = False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PrismPipeline:
    """Orchestrates the full PRISM v2.1 evaluation pipeline.

    Runs D1→D2→D3 for each problem, caches results, and provides
    accessor methods for each task's scoring data.
    """

    def __init__(
        self,
        l1_problems: list[dict],
        l2_problems: list[dict],
    ):
        self.l1_problems = l1_problems
        self.l2_problems = l2_problems

        # Separate main vs feedback problems
        self.l1_main = [p for p in l1_problems if "feedback" not in p["id"]]
        self.l1_feedback = [p for p in l1_problems if "feedback" in p["id"]]
        self.l2_main = [p for p in l2_problems if "feedback" not in p["id"]]
        self.l2_feedback = [p for p in l2_problems if "feedback" in p["id"]]

        # Result caches
        self._l1_results: list[ProblemResult] = []
        self._l2_results: list[ProblemResult] = []
        self._l1_feedback_results: list[FeedbackRoundResult] = []
        self._l2_feedback_results: list[FeedbackRoundResult] = []
        self._has_run = False

        # Task score cache — avoids recomputing expensive judge calls.
        # Keys like "t4_l1", "t4_l2" store coherence scores so Task 6
        # can read them instead of re-invoking the LLM judge.
        self._task_score_cache: dict[str, float] = {}

    # ----- Core pipeline execution -----

    def run_problem(self, llm, problem: dict, kbench) -> ProblemResult:
        """Run the full D1→D2→D3 pipeline for a single problem.

        Uses kbench chat management for multi-turn conversation.
        """
        from prism_v2.prompts.system import SYSTEM_PROMPT
        from prism_v2.prompts.prospective import build_prospective_prompt
        from prism_v2.prompts.solve import build_solve_prompt
        from prism_v2.prompts.retrospective import build_retrospective_prompt
        from prism_v2.scoring.confidence_parser import (
            parse_prospective,
            parse_retrospective,
        )
        from prism_v2.scoring.step_scorer import (
            extract_step_answers,
            extract_final_answer,
            score_steps,
        )

        result = ProblemResult(
            problem_id=problem["id"],
            novelty_level=problem["novelty_level"],
            problem_type=problem["problem_type"],
            difficulty=problem["difficulty"],
            num_steps=problem["num_steps"],
        )

        num_steps = problem["num_steps"]
        gt_steps = problem["ground_truth_steps"]

        with kbench.chats.new(
            f"prism_{problem['id']}",
            system_instructions=SYSTEM_PROMPT,
        ):
            # --- Dimension 1: Prospective ---
            d1_prompt = build_prospective_prompt(
                problem["problem_statement"], num_steps
            )
            d1_response = llm.prompt(d1_prompt)

            d1_report = parse_prospective(str(d1_response), num_steps)
            result.d1_predicted_weakest = d1_report.predicted_weakest_step
            result.d1_confidence_vector = d1_report.confidence_vector
            result.d1_confidence_labels = d1_report.confidence_labels
            result.d1_bet_correct = d1_report.bet_fraction_correct or 0.5
            result.parse_errors.extend(d1_report.parse_errors)

            # --- Dimension 2: Performance ---
            d2_prompt = build_solve_prompt()
            d2_response = llm.prompt(d2_prompt)

            d2_text = str(d2_response)
            result.d2_step_answers = extract_step_answers(d2_text, num_steps)
            result.d2_final_answer = extract_final_answer(d2_text)
            result.d2_step_correct = score_steps(result.d2_step_answers, gt_steps)
            result.d2_overall_correct = all(result.d2_step_correct)

            # Determine actual weakest step (first incorrect, or None)
            incorrect_steps = [
                i + 1 for i, c in enumerate(result.d2_step_correct) if not c
            ]
            result.d2_actual_weakest = incorrect_steps[0] if incorrect_steps else None

            # --- Dimension 3: Retrospective ---
            d3_prompt = build_retrospective_prompt(
                result.d2_step_correct,
                result.d2_overall_correct,
            )
            d3_response = llm.prompt(d3_prompt)

            d3_report = parse_retrospective(str(d3_response), num_steps)
            result.d3_reported_hardest = d3_report.reported_hardest_step
            result.d3_self_assessment = d3_report.self_assessment
            result.d3_counterfactual = d3_report.counterfactual_text
            result.parse_errors.extend(d3_report.parse_errors)

        return result

    def run_feedback_rounds(
        self,
        llm,
        feedback_problems: list[dict],
        kbench,
        num_rounds: int = 5,
    ) -> list[FeedbackRoundResult]:
        """Run feedback rounds for adaptive calibration (Task 5).

        Uses the same conversation across rounds so the model has context.
        """
        from prism_v2.prompts.system import SYSTEM_PROMPT
        from prism_v2.prompts.prospective import build_prospective_prompt
        from prism_v2.prompts.solve import build_solve_prompt
        from prism_v2.prompts.feedback import build_feedback_prompt
        from prism_v2.scoring.confidence_parser import (
            parse_prospective,
            parse_feedback_round,
        )
        from prism_v2.scoring.step_scorer import (
            extract_step_answers,
            extract_final_answer,
            score_steps,
        )

        results = []
        actual_rounds = min(num_rounds, len(feedback_problems))

        with kbench.chats.new("prism_feedback", system_instructions=SYSTEM_PROMPT):
            prev_step_results = None
            prev_confidence_labels = None

            for round_num in range(actual_rounds):
                problem = feedback_problems[round_num]
                num_steps = problem["num_steps"]
                gt_steps = problem["ground_truth_steps"]

                rr = FeedbackRoundResult(
                    problem_id=problem["id"],
                    round_number=round_num + 1,
                )

                if round_num == 0:
                    # First round: standard prospective + solve
                    d1_prompt = build_prospective_prompt(
                        problem["problem_statement"], num_steps
                    )
                    d1_resp = llm.prompt(d1_prompt)
                    d1_report = parse_prospective(str(d1_resp), num_steps)

                    rr.d1_confidence_vector = d1_report.confidence_vector
                    rr.d1_confidence_labels = d1_report.confidence_labels
                    rr.d1_bet_correct = d1_report.bet_fraction_correct or 0.5

                    solve_resp = llm.prompt(build_solve_prompt())
                    solve_text = str(solve_resp)
                    step_answers = extract_step_answers(solve_text, num_steps)
                    rr.d2_step_correct = score_steps(step_answers, gt_steps)
                    rr.d2_overall_correct = all(rr.d2_step_correct)
                else:
                    # Subsequent rounds: combined feedback + solve
                    fb_prompt = build_feedback_prompt(
                        prev_step_results,
                        prev_confidence_labels,
                        problem["problem_statement"],
                        num_steps,
                    )
                    fb_resp = llm.prompt(fb_prompt)
                    fb_text = str(fb_resp)

                    fb_report = parse_feedback_round(fb_text, num_steps)
                    rr.d1_confidence_vector = fb_report.prospective.confidence_vector
                    rr.d1_confidence_labels = fb_report.prospective.confidence_labels
                    rr.d1_bet_correct = (
                        fb_report.prospective.bet_fraction_correct or 0.5
                    )
                    rr.d2_step_correct = score_steps(fb_report.step_answers, gt_steps)
                    rr.d2_overall_correct = all(rr.d2_step_correct)

                # Store for next round
                prev_step_results = rr.d2_step_correct
                prev_confidence_labels = rr.d1_confidence_labels

                results.append(rr)

        return results

    def run_all(self, llm, kbench):
        """Run the complete pipeline for all problems (both novelty levels).

        This is the main entry point. Populates the result caches.
        """
        if self._has_run:
            return  # Already ran; use cached results

        # L1 main problems
        for problem in self.l1_main:
            result = self.run_problem(llm, problem, kbench)
            self._l1_results.append(result)

        # L2 main problems
        for problem in self.l2_main:
            result = self.run_problem(llm, problem, kbench)
            self._l2_results.append(result)

        # L1 feedback rounds
        if self.l1_feedback:
            self._l1_feedback_results = self.run_feedback_rounds(
                llm, self.l1_feedback, kbench
            )

        # L2 feedback rounds
        if self.l2_feedback:
            self._l2_feedback_results = self.run_feedback_rounds(
                llm, self.l2_feedback, kbench
            )

        self._has_run = True

    # ----- Data accessors for scoring -----

    def get_results(self, novelty_level: int = 1) -> list[ProblemResult]:
        """Get main problem results for a novelty level."""
        return self._l1_results if novelty_level == 1 else self._l2_results

    def get_feedback_results(self, novelty_level: int = 1) -> list[FeedbackRoundResult]:
        """Get feedback round results for a novelty level."""
        return (
            self._l1_feedback_results
            if novelty_level == 1
            else self._l2_feedback_results
        )

    def get_bet_fractions_and_outcomes(
        self, novelty_level: int = 1
    ) -> tuple[list[float], list[int]]:
        """Get (confidences, outcomes) pairs for AUROC computation."""
        results = self.get_results(novelty_level)
        bets = [r.d1_bet_correct for r in results]
        outcomes = [1 if r.d2_overall_correct else 0 for r in results]
        return bets, outcomes

    def get_step_rhos(self, novelty_level: int = 1) -> list:
        """Get per-problem Spearman rho values for Task 2."""
        from prism_v2.scoring.metrics import compute_spearman_rho

        results = self.get_results(novelty_level)
        rhos = []
        for r in results:
            rho = compute_spearman_rho(
                [float(c) for c in r.d1_confidence_vector],
                [1.0 if c else 0.0 for c in r.d2_step_correct],
            )
            rhos.append(rho)
        return rhos

    def get_retro_data(self, novelty_level: int = 1) -> tuple:
        """Get data needed for Task 3 (retrospective accuracy)."""
        results = self.get_results(novelty_level)
        retro_assessments = [r.d3_self_assessment for r in results]
        step_correctness = [r.d2_step_correct for r in results]
        prospective_confidences = [r.d1_confidence_vector for r in results]
        return retro_assessments, step_correctness, prospective_confidences

    def get_coherence_data(self, novelty_level: int = 1) -> dict:
        """Get data needed for Task 4 (coherence)."""
        results = self.get_results(novelty_level)
        return {
            "predicted_weakest": [r.d1_predicted_weakest for r in results],
            "reported_hardest": [r.d3_reported_hardest for r in results],
            "prospective_vectors": [r.d1_confidence_vector for r in results],
            "retro_assessments": [r.d3_self_assessment for r in results],
            "counterfactuals": [r.d3_counterfactual for r in results],
            "problem_statements": [
                next(
                    (
                        p["problem_statement"]
                        for p in (self.l1_main if novelty_level == 1 else self.l2_main)
                        if p["id"] == r.problem_id
                    ),
                    "",
                )
                for r in results
            ],
            "solve_responses": [r.d2_final_answer for r in results],
        }

    def get_feedback_data(self, novelty_level: int = 1) -> tuple:
        """Get data needed for Task 5 (adaptive calibration)."""
        results = self.get_feedback_results(novelty_level)
        round_confs = [r.d1_confidence_vector for r in results]
        round_corrs = [r.d2_step_correct for r in results]
        return round_confs, round_corrs

    def get_parse_error_rate(self) -> float:
        """Compute the overall parse error rate across all results."""
        all_results = self._l1_results + self._l2_results
        if not all_results:
            return 0.0
        problems_with_errors = sum(1 for r in all_results if r.parse_errors)
        return problems_with_errors / len(all_results)

    def get_counterfactual_parse_rate(self) -> float:
        """Compute the rate of non-empty counterfactual responses.

        Returns the proportion of main problems where the model produced
        a non-empty COUNTERFACTUAL: response. Low rates indicate the model
        failed to follow the retrospective prompt format.
        """
        all_results = self._l1_results + self._l2_results
        if not all_results:
            return 0.0
        non_empty = sum(
            1
            for r in all_results
            if r.d3_counterfactual and r.d3_counterfactual.strip()
        )
        return non_empty / len(all_results)

    def summary(self) -> dict:
        """Return a summary of the pipeline run."""
        return {
            "l1_main_count": len(self._l1_results),
            "l2_main_count": len(self._l2_results),
            "l1_feedback_rounds": len(self._l1_feedback_results),
            "l2_feedback_rounds": len(self._l2_feedback_results),
            "parse_error_rate": self.get_parse_error_rate(),
            "l1_overall_accuracy": (
                sum(1 for r in self._l1_results if r.d2_overall_correct)
                / len(self._l1_results)
                if self._l1_results
                else 0.0
            ),
            "l2_overall_accuracy": (
                sum(1 for r in self._l2_results if r.d2_overall_correct)
                / len(self._l2_results)
                if self._l2_results
                else 0.0
            ),
        }
