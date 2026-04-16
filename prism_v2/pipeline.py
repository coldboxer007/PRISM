"""
PRISM v2.1 Pipeline Orchestrator

Runs the full D1→D2→D3a→D3b metacognitive assessment pipeline for each problem.
Implements result caching so multiple tasks can share a single pipeline run.

D3a is a *blind* retrospective turn: the model assesses its own step correctness
BEFORE seeing results.  D3b is the *informed* turn: results are revealed, and the
model identifies the hardest step and provides counterfactual analysis.

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
    d2_final_correct: bool = False
    d2_solve_text: str = ""  # full reasoning trace for counterfactual judge

    # Dimension 3a: Blind Retrospective (before results revealed)
    d3a_blind_assessment: list[str] = field(default_factory=list)

    # Dimension 3b: Informed Retrospective (after results revealed)
    d3_reported_hardest: Optional[int] = None
    d3_counterfactual: str = ""

    # Metadata
    parse_errors: list[str] = field(default_factory=list)
    num_steps: int = 5


@dataclass
class DecisionResult:
    """Result for a single metacognitive decision problem (Task 5)."""

    problem_id: str
    novelty_level: int
    is_solvable: bool

    # Model's decision
    decision: str = "accept"  # "accept" or "decline"
    confidence: int = 50  # 0-100
    reasoning: str = ""

    # Solve attempt (always attempted regardless of decision)
    model_correct: bool = False
    model_answer: str = ""

    # Payoff structure
    payoff_correct: int = 0
    payoff_wrong: int = 0
    payoff_decline: int = 0

    # Parse metadata
    parse_errors: list[str] = field(default_factory=list)


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

        # Separate main vs decision problems
        self.l1_main = [p for p in l1_problems if "decision" not in p["id"]]
        self.l1_decision = [p for p in l1_problems if "decision" in p["id"]]
        self.l2_main = [p for p in l2_problems if "decision" not in p["id"]]
        self.l2_decision = [p for p in l2_problems if "decision" in p["id"]]

        # Result caches
        self._l1_results: list[ProblemResult] = []
        self._l2_results: list[ProblemResult] = []
        self._l1_decision_results: list[DecisionResult] = []
        self._l2_decision_results: list[DecisionResult] = []
        self._has_run = False

        # Task score cache — avoids recomputing expensive judge calls.
        # Keys like "t4_l1", "t4_l2" store coherence scores so Task 6
        # can read them instead of re-invoking the LLM judge.
        self._task_score_cache: dict[str, float] = {}

    # ----- Core pipeline execution -----

    def run_problem(self, llm, problem: dict, kbench) -> ProblemResult:
        """Run the full D1→D2→D3a→D3b pipeline for a single problem.

        Uses kbench chat management for multi-turn conversation.
        D3a is a blind self-assessment (before results); D3b reveals
        results and asks for hardest step + counterfactual.
        """
        from prism_v2.prompts.system import SYSTEM_PROMPT
        from prism_v2.prompts.prospective import build_prospective_prompt
        from prism_v2.prompts.solve import build_solve_prompt
        from prism_v2.prompts.retrospective import (
            build_blind_retrospective_prompt,
            build_informed_retrospective_prompt,
        )
        from prism_v2.scoring.confidence_parser import (
            parse_prospective,
            parse_blind_retrospective,
            parse_retrospective,
        )
        from prism_v2.scoring.step_scorer import (
            extract_step_answers,
            extract_final_answer,
            compare_answers,
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
            result.d2_solve_text = d2_text  # store full trace for judge
            result.d2_step_answers = extract_step_answers(d2_text, num_steps)
            result.d2_final_answer = extract_final_answer(d2_text)
            result.d2_step_correct = score_steps(result.d2_step_answers, gt_steps)
            result.d2_final_correct = compare_answers(
                result.d2_final_answer,
                problem["ground_truth_final"],
            )
            result.d2_overall_correct = (
                all(result.d2_step_correct) and result.d2_final_correct
            )

            # Determine actual weakest step (first incorrect, or None)
            incorrect_steps = [
                i + 1 for i, c in enumerate(result.d2_step_correct) if not c
            ]
            result.d2_actual_weakest = incorrect_steps[0] if incorrect_steps else None

            # --- Dimension 3a: Blind Retrospective ---
            # Model assesses its own steps BEFORE seeing results.
            d3a_prompt = build_blind_retrospective_prompt(num_steps)
            d3a_response = llm.prompt(d3a_prompt)

            d3a_report = parse_blind_retrospective(str(d3a_response), num_steps)
            result.d3a_blind_assessment = d3a_report.self_assessment
            result.parse_errors.extend(d3a_report.parse_errors)

            # --- Dimension 3b: Informed Retrospective ---
            # Reveal results, ask for hardest step + counterfactual.
            d3b_prompt = build_informed_retrospective_prompt(
                result.d2_step_correct,
                result.d2_overall_correct,
            )
            d3b_response = llm.prompt(d3b_prompt)

            d3b_report = parse_retrospective(str(d3b_response), num_steps)
            result.d3_reported_hardest = d3b_report.reported_hardest_step
            result.d3_counterfactual = d3b_report.counterfactual_text
            result.parse_errors.extend(d3b_report.parse_errors)

        return result

    def run_decision_problem(self, llm, problem: dict, kbench) -> "DecisionResult":
        """Run a single metacognitive decision problem.

        Uses a single chat turn: the decision prompt asks the model to
        (1) decide ACCEPT/DECLINE, then (2) attempt the problem regardless.
        We parse both the decision and the solve attempt.
        """
        from prism_v2.prompts.system import SYSTEM_PROMPT
        from prism_v2.prompts.decision import build_decision_prompt
        from prism_v2.scoring.decision_scorer import parse_decision_response
        from prism_v2.scoring.step_scorer import extract_final_answer, compare_answers

        is_solvable = problem.get("is_solvable", True)

        dr = DecisionResult(
            problem_id=problem["id"],
            novelty_level=problem["novelty_level"],
            is_solvable=is_solvable,
            payoff_correct=problem["payoff_correct"],
            payoff_wrong=problem["payoff_wrong"],
            payoff_decline=problem["payoff_decline"],
        )

        with kbench.chats.new(
            f"prism_{problem['id']}",
            system_instructions=SYSTEM_PROMPT,
        ):
            prompt = build_decision_prompt(problem)
            response = llm.prompt(prompt)
            resp_text = str(response)

        # Parse the decision
        parsed = parse_decision_response(resp_text)
        dr.decision = parsed.decision
        dr.confidence = parsed.confidence
        dr.reasoning = parsed.reasoning
        dr.parse_errors = parsed.parse_errors

        # Check correctness of the solve attempt
        if is_solvable:
            model_final = extract_final_answer(resp_text)
            dr.model_answer = model_final
            gt = problem.get("ground_truth_final", "")
            dr.model_correct = compare_answers(model_final, gt)
        else:
            # Unsolvable problems: model is always wrong
            dr.model_correct = False
            dr.model_answer = extract_final_answer(resp_text)

        return dr

    def run_decision_problems(
        self,
        llm,
        decision_problems: list[dict],
        kbench,
    ) -> list["DecisionResult"]:
        """Run all decision problems for one novelty level."""
        results = []
        for problem in decision_problems:
            dr = self.run_decision_problem(llm, problem, kbench)
            results.append(dr)
        return results

    def run_all(self, llm, kbench):
        """Run the complete pipeline for all problems (both novelty levels).

        This is the main entry point. Populates the result caches.
        Runs main problems (D1→D2→D3a→D3b) and decision problems (Task 5).
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

        # L1 decision problems (Task 5)
        if self.l1_decision:
            self._l1_decision_results = self.run_decision_problems(
                llm, self.l1_decision, kbench
            )

        # L2 decision problems (Task 5)
        if self.l2_decision:
            self._l2_decision_results = self.run_decision_problems(
                llm, self.l2_decision, kbench
            )

        self._has_run = True

    # ----- Data accessors for scoring -----

    def get_results(self, novelty_level: int = 1) -> list[ProblemResult]:
        """Get main problem results for a novelty level."""
        return self._l1_results if novelty_level == 1 else self._l2_results

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
        """Get data needed for Task 3 (retrospective accuracy).

        Returns blind (D3a) assessments — the model's self-assessment
        BEFORE seeing which steps were correct/incorrect.
        """
        results = self.get_results(novelty_level)
        blind_assessments = [r.d3a_blind_assessment for r in results]
        step_correctness = [r.d2_step_correct for r in results]
        prospective_confidences = [r.d1_confidence_vector for r in results]
        return blind_assessments, step_correctness, prospective_confidences

    def get_coherence_data(self, novelty_level: int = 1) -> dict:
        """Get data needed for Task 4 (coherence).

        Sub-score A uses D3b reported_hardest (informed — model knows results).
        Sub-score B uses D3a blind_assessment (model's own judgment, no leaking).
        Sub-score C uses D3b counterfactual (informed — needs result context).
        """
        results = self.get_results(novelty_level)
        return {
            "predicted_weakest": [r.d1_predicted_weakest for r in results],
            "reported_hardest": [r.d3_reported_hardest for r in results],
            "prospective_vectors": [r.d1_confidence_vector for r in results],
            "retro_assessments": [r.d3a_blind_assessment for r in results],
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
            "solve_responses": [r.d2_solve_text for r in results],
        }

    def get_decision_results(self, novelty_level: int = 1) -> list["DecisionResult"]:
        """Get decision problem results for a novelty level."""
        return (
            self._l1_decision_results
            if novelty_level == 1
            else self._l2_decision_results
        )

    def get_decision_data(self, novelty_level: int = 1) -> dict:
        """Get data needed for Task 5 (metacognitive control).

        Returns dict with parallel lists ready for compute_metacognitive_control().
        """
        results = self.get_decision_results(novelty_level)
        return {
            "decisions": [r.decision for r in results],
            "is_solvable": [r.is_solvable for r in results],
            "model_correct": [r.model_correct for r in results],
            "payoff_correct": [r.payoff_correct for r in results],
            "payoff_wrong": [r.payoff_wrong for r in results],
            "payoff_decline": [r.payoff_decline for r in results],
        }

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
            "l1_decision_count": len(self._l1_decision_results),
            "l2_decision_count": len(self._l2_decision_results),
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
