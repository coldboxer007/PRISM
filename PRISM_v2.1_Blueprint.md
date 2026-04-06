# PRISM v2.1: Prospective-Retrospective Introspective Self-Model

## Technical Blueprint for Implementation

**Target:** Kaggle "Measuring Progress Toward AGI — Cognitive Abilities" Hackathon  
**Track:** Metacognition  
**Deadline:** April 16, 2026  
**Platform:** Kaggle Community Benchmarks (kaggle-benchmarks SDK)

---

## 1. What PRISM v2.1 Is

PRISM v2.1 is a metacognition benchmark that measures whether LLMs can accurately monitor their own reasoning — before, during, and after solving problems. It operationalizes the Nelson & Narens (1990) monitoring/control framework: the gold-standard cognitive science model of metacognition that DeepMind's own taxonomy references.

The core question: **Does the model know where it will struggle, does it know where it actually struggled, and do those two stories agree?**

Most existing metacognition benchmarks measure calibration — does stated confidence match accuracy? PRISM goes three levels deeper:

1. **Step-level resolution** — not just "how confident are you?" but "which specific step will fail?"
2. **Temporal coherence** — do pre-task predictions and post-task reflections tell the same story?
3. **Adaptive learning** — does the model update beliefs appropriately after feedback?

No existing benchmark tests all three simultaneously.

---

## 2. Why This Design Wins

### 2.1 What Judges Are Looking For (from competition rubric)

| Criterion (Weight) | How PRISM Addresses It |
|---|---|
| Dataset quality & task construction (50%) | Objectively verifiable multi-step math problems. Each step has a deterministic correct intermediate answer. No ambiguity. |
| Writeup quality (20%) | Grounded in established cognitive science (Nelson & Narens 1990, Maniscalco & Lau 2012, Fleming 2017). Clear methodology. |
| Novelty, insights, discriminatory power (30%) | Prospective-retrospective coherence is entirely novel. Step-level metacognition is novel. The framework produces a gradient — not all-pass or all-fail. |

### 2.2 What Competing Submissions Probably Look Like

Based on the competition description and typical approaches:

- **Basic calibration:** Ask confidence, check against accuracy. Produces ECE. This is table stakes — Cacioli (2026) already did this rigorously.
- **"Are you sure?" probes:** Ask model if it wants to change its answer. Shallow.
- **Knowledge boundary tests:** "Do you know X?" type questions. Tests knowledge, not metacognitive monitoring.

PRISM v2.1 is differentiated because it tests the **internal consistency of metacognitive reports across time** — something no published benchmark does.

### 2.3 What the DeepMind Paper Specifically Asks For

From the cognitive taxonomy paper, metacognition includes:
- **Metacognitive knowledge:** Understanding of one's own cognitive processes
- **Metacognitive monitoring:** Tracking ongoing cognitive performance
- **Metacognitive control:** Adjusting strategies based on monitoring

PRISM tests all three:
- Knowledge → the model must predict which step will be hard (requires self-knowledge)
- Monitoring → the model must assess per-step outcomes retrospectively
- Control → the model must update confidence after feedback (adaptive calibration)

---

## 3. Architecture Overview

### 3.1 The Three Dimensions

```
DIMENSION 1: PROSPECTIVE REPORT    →  "Where will I struggle?"
     ↓
DIMENSION 2: PERFORMANCE           →  Actually solve the problem
     ↓  
DIMENSION 3: RETROSPECTIVE REPORT  →  "Where did I actually struggle?"
```

Each dimension produces structured data. The gaps between dimensions are the metacognitive scores.

### 3.2 The Two Novelty Levels

- **L1 (Familiar):** Standard multi-step problems likely encountered in training data
- **L2 (Novel):** Problems using in-context novel rule systems with identical structure

The L1-to-L2 slope is the contamination control. If metacognitive ability drops sharply on novel tasks, the model was pattern-matching, not genuinely self-monitoring.

### 3.3 Metacognitive Control (Decision Problems)

Five decision problems per novelty level. The model evaluates each problem and decides whether to ACCEPT or DECLINE it, given an explicit payoff structure (+X for correct, -Y for wrong, +Z for decline). Two of the five problems are intentionally unsolvable (contradictory system, missing-info word problem). This tests metacognitive control — the ability to evaluate one's own competence before committing to a task.

### 3.4 The Six Benchmark Tasks

Each task in the Kaggle Benchmark tests one specific metacognitive ability and returns a score between 0.0 and 1.0.

1. **Prospective Calibration** (AUROC of bet-fraction confidence)
2. **Step-Level Prospective Accuracy** (Spearman ρ, confidence vs. correctness)
3. **Retrospective Self-Assessment Accuracy** (proportion of correct self-assessments)
4. **Prospective-Retrospective Coherence** (composite consistency score)
5. **Metacognitive Control** (accept/decline utility ratio on decision problems)
6. **Novelty Robustness** (L2/L1 metacognitive score ratio)

---

## 4. Problem Design

This is the most critical design decision. Problems must satisfy ALL of these constraints:

1. **Multi-step with objectively verifiable intermediate answers** — so we can score per-step correctness without subjective judgment
2. **Naturally numbered steps** — so we can ask "which step?" without the model inventing arbitrary decompositions
3. **Variable difficulty** — so calibration has a signal to track
4. **Novel variants possible** — so we can create L2 versions
5. **Compact enough to fit in quota** — the full prompt + response must be reasonable

### 4.1 Problem Domain: Multi-Step Mathematical Derivations

The best fit is **systems of equations and multi-step algebraic/arithmetic derivations** where each step produces a concrete intermediate value.

#### L1 (Familiar) Problem Template

Problems that LLMs have definitely seen variants of:

**Type A — Systems of Linear Equations (3 variables, 5 steps):**
- Step 1: Use equation 1 to express variable x in terms of y and z
- Step 2: Substitute into equation 2 to eliminate x
- Step 3: Use result to express y in terms of z
- Step 4: Substitute into equation 3 to solve for z
- Step 5: Back-substitute to find x and y

Each step produces a specific algebraic expression or numeric value that is deterministically correct or incorrect.

**Type B — Multi-Step Word Problems (5 steps):**
- Step 1: Extract and set up the relevant equation
- Step 2: Compute intermediate value A
- Step 3: Apply constraint to get intermediate value B
- Step 4: Combine A and B
- Step 5: Compute final answer and verify

**Type C — Modular Arithmetic / Number Theory (5 steps):**
- Problems involving successive applications of modular operations
- Each step produces a specific numeric residue

#### L2 (Novel) Problem Template

Same structural skeleton, but with in-context rule modifications:

**Type A-Novel — Systems with Custom Operators:**
"In this problem, we use Zeta-addition defined as: a ⊕ b = a + b + 1. Zeta-multiplication is defined as: a ⊗ b = a × b − 1. Solve the following system using only Zeta-operations..."

The model must learn the novel operators in-context and apply them correctly. The steps are identical in structure — only the arithmetic rules change.

**Type B-Novel — Word Problems with Inverted Logic:**
"In Mirrorland, prices decrease when demand increases. Every unit of demand increase reduces price by the demand-price coefficient. Given..."

**Type C-Novel — Non-Standard Modular Systems:**
"In base-7 residue arithmetic where the modular inverse operation is defined as..."

#### 4.2 Problem Generation Strategy

We need a dataset of problems. The approach:

1. **Create a problem generator** — a Python function that takes parameters (number of variables, coefficient ranges, operator definitions) and produces:
   - The problem statement
   - The correct answer for each step
   - The final answer
   - Difficulty metadata (coefficient size, number of carries, etc.)

2. **Generate L1 problems** with standard arithmetic. Vary difficulty by:
   - Coefficient magnitude (small integers → larger integers → fractions)
   - Number of steps that require carrying/borrowing
   - Whether intermediate results are "clean" numbers or messy fractions

3. **Generate L2 problems** from the same generator by swapping in novel operator definitions. The structural skeleton is identical — only the operator semantics change.

4. **Verify all problems** by running through stdlib arithmetic (`int`, `float`, `pow`) to confirm every intermediate step has a unique correct answer.

#### 4.3 Problem Count

Given quota constraints ($50/day, $500/month), we need to be efficient:

- Each main problem requires 3 API calls (prospective, solve, retrospective)
- Each decision problem requires 1 API call (combined decision + solve)
- Target: **10 main problems per novelty level** (Tasks 1-4, 6)
- Target: **5 decision problems per novelty level** (Task 5)
- Total per model: 10×3×2 + 5×1×2 = **70 API calls**

This is feasible within daily quota if we batch efficiently.

#### 4.4 Step Scoring

For each problem, we need to verify per-step correctness. The approach:

1. The problem generator produces a **ground truth array** of intermediate answers: one per step
2. When the model solves the problem, we instruct it to show work step-by-step and box each intermediate answer
3. We extract each intermediate answer (via regex or structured output) and compare against ground truth
4. Each step gets a binary score: correct (1) or incorrect (0)
5. If a step is incorrect, subsequent steps that depend on it are scored based on whether they correctly applied the (wrong) intermediate result — this is important because carrying forward a wrong value correctly is still evidence of reasoning ability, even if the answer is wrong

**Decision on error propagation scoring:** For simplicity in v2.1, we score each step purely against the ground truth. A model that makes an error in Step 2 but correctly executes Steps 3-5 using the wrong Step 2 result will still score Steps 3-5 as incorrect. This is conservative but avoids the complexity of partial-credit scoring for the initial submission.

---

## 5. Prompt Design

This section specifies the exact prompts used in each dimension. These are the most sensitive component — small wording changes affect model behavior significantly.

### 5.1 System Prompt (Used Across All Interactions)

```
You are solving multi-step mathematical problems. You will be asked 
to assess your own confidence and reasoning at various points. 

When solving problems, show your work for each step clearly and 
label each step with its number (Step 1, Step 2, etc.). Box or 
clearly mark each intermediate answer.

When rating your confidence, be as honest and accurate as possible 
about your actual uncertainty — do not default to expressing high 
confidence unless you genuinely feel certain.
```

**Rationale:** The instruction to "be honest" has been shown to slightly improve calibration in some models (Tian et al. 2023). The structured output request ensures we can parse step-level answers.

### 5.2 Dimension 1: Prospective Metacognitive Report

This is delivered BEFORE the model sees the problem solution task. The model sees the problem statement but is asked to reflect before solving.

```
I'm going to ask you to solve a multi-step math problem. Before you 
solve it, I want you to assess your own capabilities on this specific 
problem.

Here is the problem:
[PROBLEM STATEMENT]

This problem has [N] steps. Before solving it, answer these three 
questions:

1. WEAKEST STEP: Which step number (1 through [N]) do you predict 
   will be the most difficult or error-prone for you? Give just the 
   number.

2. PER-STEP CONFIDENCE: For each step, rate your confidence using 
   exactly one of these labels:
   - "definitely right" 
   - "probably right"
   - "uncertain"
   - "probably wrong"
   - "definitely wrong"
   
   Format as:
   Step 1: [label]
   Step 2: [label]
   ... etc.

3. OVERALL BET: If you had to bet $100 on whether your final answer 
   will be correct, how would you split it? State the amount you'd 
   bet on CORRECT and the amount on INCORRECT. They must sum to $100.

Answer ONLY these three questions. Do not solve the problem yet.
```

**Data extraction from this response:**
- `predicted_weakest_step`: integer (1 to N)
- `prospective_confidence_vector`: array of N ordinal values, mapped as: "definitely right" = 5, "probably right" = 4, "uncertain" = 3, "probably wrong" = 2, "definitely wrong" = 1
- `bet_fraction_correct`: float between 0.0 and 1.0 (amount bet on correct / 100)

**Parsing strategy:**
- Use structured output if the SDK supports it (request a dataclass/pydantic response)
- Fallback: regex extraction. The format is constrained enough for reliable parsing.
  - Weakest step: extract first standalone integer after "WEAKEST STEP"
  - Per-step confidence: match pattern `Step \d+:\s*(definitely right|probably right|uncertain|probably wrong|definitely wrong)`
  - Bet: match pattern `\$?(\d+)\s*(on correct|on CORRECT|correct)`

### 5.3 Dimension 2: Performance (Problem Solving)

Delivered as a NEW message in the same conversation (multi-turn), so the model has context of its prospective report.

```
Now solve the problem. Show your work clearly for each step. 
Label each step (Step 1, Step 2, etc.) and clearly state each 
intermediate answer.

After all steps, state your FINAL ANSWER clearly.
```

**Data extraction:**
- `step_answers`: array of N extracted intermediate answers (strings or numbers)
- `final_answer`: the stated final answer
- `step_correctness`: array of N binary values (1 = matches ground truth, 0 = doesn't)
- `overall_correct`: binary (1 if final answer matches ground truth)

**Parsing strategy:**
- Extract intermediate answers using pattern: `Step \d+:.*?(?:answer|result|gives|=)\s*(.+?)(?:\n|$)`
- Compare extracted answers against ground truth numerically (with tolerance for floating point)
- For algebraic expressions, normalize and compare symbolically if possible, otherwise fall back to string comparison

### 5.4 Dimension 3: Retrospective Metacognitive Report

Delivered as a NEW message in the same conversation. The model is told ONLY whether it got the overall answer right or wrong, and which steps were right/wrong. It is NOT given the correct answers.

```
Here is your result: You got [X] out of [N] steps correct.
Specifically:
Step 1: [CORRECT/INCORRECT]
Step 2: [CORRECT/INCORRECT]
... etc.
Your final answer was [CORRECT/INCORRECT].

Now reflect on your performance:

1. HARDEST STEP: Which step actually gave you the most trouble 
   during your reasoning? Give just the number.

2. PER-STEP SELF-ASSESSMENT: For each step, categorize your 
   experience as exactly one of:
   - "confident and correct"
   - "confident but wrong"  
   - "uncertain and correct"
   - "uncertain and wrong"
   
   Format as:
   Step 1: [label]
   Step 2: [label]
   ... etc.

3. COUNTERFACTUAL: Name one specific thing you considered doing 
   differently but chose not to. What was the alternative approach, 
   and why did you reject it?
```

**Data extraction:**
- `reported_weakest_step`: integer (1 to N)
- `retrospective_assessment_vector`: array of N categorical values from the 2×2 grid
- `counterfactual_action`: free text describing what the model "almost did"
- `counterfactual_reason`: free text describing why it was rejected

**Parsing strategy:**
- Weakest step: same regex as D1
- Per-step assessment: match `Step \d+:\s*(confident and correct|confident but wrong|uncertain and correct|uncertain and wrong)`
- Counterfactual: extract full text after "COUNTERFACTUAL:" — this doesn't need structured parsing since it's judged by LLM-as-judge

### 5.5 Decision Problem Prompt (Task 5: Metacognitive Control)

For each decision problem the model receives the problem statement, explicit payoff structure, and instructions to first decide ACCEPT or DECLINE, then attempt the problem regardless:

```
You are given a mathematical problem with an explicit payoff structure.

PAYOFF STRUCTURE:
- If you ACCEPT and answer correctly: +X points
- If you ACCEPT and answer incorrectly: -Y points
- If you DECLINE: +Z points

IMPORTANT: Some problems may be unsolvable (contradictory or missing information).
If you believe a problem is unsolvable, DECLINE is the safe choice.

INSTRUCTIONS:
1. DECISION: State ACCEPT or DECLINE
2. CONFIDENCE: Your confidence in solving correctly (0-100%)
3. REASONING: Brief explanation of your decision

Regardless of your decision, please attempt to solve the problem below,
showing your work step by step.

[PROBLEM STATEMENT]
```

**Design note:** The model always attempts to solve (regardless of decision) so we can evaluate correctness and compute the oracle's optimal strategy. The payoff profiles vary across problems: low-risk (small penalty), medium-risk, and high-risk (large penalty). Two of the five problems per level are unsolvable (contradictory system, missing information).

---

## 6. Scoring Methodology

### 6.1 Task 1: Prospective Calibration (AUROC)

**What it measures:** Can the model's stated confidence discriminate between problems it gets right and problems it gets wrong?

**Calculation:**
1. Collect all (bet_fraction_correct, overall_correct) pairs across all problems in a novelty level
2. Compute AUROC: the area under the ROC curve where bet_fraction is the classifier score and overall_correct is the label
3. AUROC = 0.5 means the confidence is uninformative (chance). AUROC = 1.0 means perfect discrimination.

**Implementation notes:**
- Use sklearn.metrics.roc_auc_score or manual trapezoidal integration
- If all problems are correct or all incorrect, AUROC is undefined — handle by returning 0.5 and flagging
- Minimum 10 problems needed per novelty level for meaningful AUROC

**Why AUROC instead of ECE or M-ratio:**
- ECE requires binning, which is unreliable with small sample sizes (Guo et al. 2017 recommend 15 bins — we won't have enough data points)
- M-ratio requires log-probabilities, which API models don't expose (the Cacioli limitation)
- AUROC is a proper non-parametric measure of type-2 metacognitive sensitivity (Maniscalco & Lau 2012) and works with verbalized confidence

**Score for benchmark:** AUROC value directly (range 0.0 to 1.0, chance = 0.5)

### 6.2 Task 2: Step-Level Prospective Accuracy (Spearman ρ)

**What it measures:** Does the model's per-step confidence ranking predict which steps it actually gets right?

**Calculation:**
1. For each problem, we have two arrays of length N:
   - `prospective_confidence_vector` (ordinal, 1-5)
   - `step_correctness` (binary, 0 or 1)
2. Compute Spearman rank correlation between these two arrays
3. Average the per-problem Spearman ρ across all problems

**Implementation notes:**
- Use scipy.stats.spearmanr
- Handle tied ranks (which will be frequent since binary correctness has only two values) — Spearman handles this natively
- For problems where all steps are correct or all incorrect, ρ is undefined — exclude these and report the exclusion rate
- With only 5 steps per problem, individual correlations are noisy. The signal emerges from averaging across 10+ problems.

**Score for benchmark:** (mean_ρ + 1) / 2, normalized to 0-1 range. Score of 0.5 = no correlation (chance). Score of 1.0 = perfect prediction.

### 6.3 Task 3: Retrospective Self-Assessment Accuracy

**What it measures:** After being told which steps were right/wrong, can the model accurately categorize its own experience as "confident and correct," "confident but wrong," etc.?

**Calculation:**
1. For each step, we know:
   - The ground truth correctness (from D2)
   - The prospective confidence (from D1, mapped to confident/uncertain using threshold: "definitely right" and "probably right" = confident; "uncertain", "probably wrong", "definitely wrong" = uncertain)
   - The retrospective self-assessment (from D3)
2. The "correct" retrospective label is determined by combining ground truth correctness with prospective confidence:
   - If Step was correct AND prospective confidence was high → ground truth = "confident and correct"
   - If Step was correct AND prospective confidence was low → ground truth = "uncertain and correct"
   - If Step was incorrect AND prospective confidence was high → ground truth = "confident but wrong"
   - If Step was incorrect AND prospective confidence was low → ground truth = "uncertain and wrong"
3. Score = proportion of steps where retrospective label matches the ground truth label

**Why this scoring works:** This is deliberately tricky. The model must remember what it said prospectively AND correctly assess what actually happened. Models that always say "confident and correct" will score poorly on steps they got wrong. Models that retroactively revise their confidence story will score poorly on the confidence dimension.

**Score for benchmark:** Proportion correct (0.0 to 1.0).

### 6.4 Task 4: Prospective-Retrospective Coherence

**What it measures:** Do the model's before-task and after-task metacognitive reports tell a consistent story?

This is the most novel score. It has three sub-components:

**Sub-score A — Location Consistency (weight: 0.35):**
- Does D1's predicted-weakest-step match D3's reported-weakest-step?
- Binary per problem, aggregated as proportion across all problems
- Value: 0.0 to 1.0

**Sub-score B — Confidence Consistency (weight: 0.45):**
- Spearman ρ between D1's per-step confidence vector and D3's per-step difficulty reports
- D3's self-assessment is mapped to a difficulty scale: "uncertain and wrong" = 4 (hardest), "confident but wrong" = 3, "uncertain and correct" = 2, "confident and correct" = 1 (easiest)
- Average ρ across problems, then normalize: (ρ + 1) / 2
- Value: 0.0 to 1.0

**Sub-score C — Counterfactual Plausibility (weight: 0.20):**
- Use LLM-as-judge (via kbench.assertions.assess_response_with_judge) to evaluate whether the counterfactual from D3 is plausible
- Judge criteria:
  1. "The alternative approach described is a mathematically valid method for this type of problem"
  2. "The stated reason for rejecting the alternative is logically coherent"
  3. "The alternative approach is substantively different from what was actually done (not a trivial rephrasing)"
- Score = proportion of criteria passed (0, 0.33, 0.67, or 1.0)
- Average across all problems

**Composite score:** 0.35 × A + 0.45 × B + 0.20 × C

**Score for benchmark:** Composite (0.0 to 1.0)

### 6.5 Task 5: Metacognitive Control (Accept/Decline Decisions)

**What it measures:** Can the model accurately evaluate its own competence and make rational accept/decline decisions given explicit payoff structures?

**Design:**
- 5 decision problems per novelty level:
  - 3 solvable: Type A easy/low-risk (+5/-3/+1), Type B medium/medium-risk (+10/-15/+2), Type C hard/high-risk (+20/-30/+3)
  - 2 unsolvable: contradictory system (Type A), missing-info word problem (Type B)
- The model must ACCEPT or DECLINE each problem given the payoff structure
- Regardless of decision, the model always attempts to solve (for calibration)

**Calculation:**
1. For each problem, compute the payoff based on the model's decision and actual correctness:
   - If ACCEPT: payoff = payoff_correct if correct, payoff_wrong if wrong
   - If DECLINE: payoff = payoff_decline
2. Sum all payoffs to get `model_utility`
3. Compute `optimal_utility`: the utility achieved by a perfect oracle that accepts only solvable problems it would get right, and declines everything else
4. Compute `worst_utility`: the utility from the worst possible strategy (accept all, get all wrong)
5. Score = `(model_utility - worst_utility) / (optimal_utility - worst_utility)`, clamped to [0, 1]

**Baseline note:** A "decline everything" strategy scores approximately 0.54 (not 0.0), because declining avoids the large penalties from wrong answers. This is by design — the 0.46 gap between "decline all" and perfect metacognition provides meaningful discrimination. A model must demonstrate genuine problem-difficulty evaluation to approach 1.0.

**Score for benchmark:** Utility ratio (0.0 to 1.0)

### 6.6 Task 6: Novelty Robustness

**What it measures:** Does the model's metacognitive ability survive when problems use novel, in-context rules?

**Calculation:**
1. Compute a composite metacognitive score for L1 (average of Tasks 1-4 on L1 problems)
2. Compute the same composite for L2
3. Ratio = L2_composite / L1_composite

**Interpretation:**
- Ratio ≈ 1.0 → metacognition is robust to novelty (genuine self-monitoring)
- Ratio << 1.0 → metacognition degrades on novel tasks (was pattern-matching)
- Ratio > 1.0 → metacognition is actually better on novel tasks (possible if familiarity causes overconfidence)

**Score for benchmark:** min(ratio, 1.0) — capped at 1.0

---

## 7. Implementation Architecture

### 7.1 Project File Structure

```
prism_v2/
├── tasks/
│   ├── task_01_prospective_calibration.py      # AUROC task
│   ├── task_02_step_accuracy.py                # Spearman ρ task
│   ├── task_03_retrospective_accuracy.py       # Self-assessment task
│   ├── task_04_coherence.py                    # Coherence composite task
│   ├── task_05_adaptive_calibration.py         # Feedback drift task
│   └── task_06_novelty_robustness.py           # L1/L2 ratio task
├── problems/
│   ├── generator.py                            # Problem generation engine
│   ├── l1_problems.json                        # Pre-generated L1 problem set
│   └── l2_problems.json                        # Pre-generated L2 problem set
├── scoring/
│   ├── step_scorer.py                          # Per-step answer extraction & scoring
│   ├── confidence_parser.py                    # Parse confidence responses
│   └── metrics.py                              # AUROC, Spearman, drift calculations
├── prompts/
│   ├── system.py                               # System prompt
│   ├── prospective.py                          # D1 prompt template
│   ├── solve.py                                # D2 prompt template
│   ├── retrospective.py                        # D3 prompt template
│   └── feedback.py                             # Feedback round prompt template
└── notebook.ipynb                              # Main Kaggle notebook
```

### 7.2 Execution Flow Per Problem

This is the exact sequence of operations for a single problem:

```
1. Load problem from dataset (statement, steps, ground_truth_answers)
2. Create a new chat context/session with the LLM
3. Send system prompt
4. Send D1 (prospective) prompt with problem statement
5. Parse D1 response → extract predicted_weakest, confidence_vector, bet_fraction
6. Send D2 (solve) prompt in the same conversation
7. Parse D2 response → extract per-step answers
8. Score each step against ground truth → step_correctness array
9. Construct D3 prompt with step-level results (but not correct answers)
10. Send D3 (retrospective) prompt in the same conversation
11. Parse D3 response → extract reported_weakest, self_assessment, counterfactual
12. Store all data in a results dictionary for this problem
```

For decision problems (Task 5), steps 4-12 are replaced by a single combined decision + solve prompt. The model's ACCEPT/DECLINE choice and attempted solution are parsed from one response.

### 7.3 Data Structures

**Problem record:**
```
{
  "id": "l1_003",
  "novelty_level": 1,
  "problem_statement": "Solve the system: 2x + 3y - z = 7 ...",
  "num_steps": 5,
  "step_descriptions": ["Isolate x from eq1", "Substitute into eq2", ...],
  "ground_truth_steps": ["x = (7 - 3y + z)/2", "5y - 3z = 1", ...],
  "ground_truth_final": "x=2, y=1, z=0",
  "difficulty_metadata": {"coefficient_range": [1,5], "has_fractions": false}
}
```

**Per-problem result record:**
```
{
  "problem_id": "l1_003",
  "model": "gemini-2.0-flash",
  
  "d1_predicted_weakest": 3,
  "d1_confidence_vector": [5, 4, 3, 4, 5],
  "d1_bet_correct": 0.75,
  
  "d2_step_answers": ["(7-3y+z)/2", "5y-3z=1", ...],
  "d2_step_correct": [1, 1, 0, 1, 0],
  "d2_overall_correct": 0,
  "d2_actual_weakest": 3,
  
  "d3_reported_weakest": 3,
  "d3_self_assessment": ["confident and correct", "confident and correct", 
                          "uncertain and wrong", "uncertain and correct", 
                          "confident but wrong"],
  "d3_counterfactual": "I considered using Cramer's rule instead of substitution...",
  "d3_counterfactual_reason": "I rejected it because the determinant computation seemed more error-prone..."
}
```

**Decision result record:**
```
{
  "problem_id": "l1_decision_001",
  "decision": "accept",
  "confidence": 85,
  "reasoning": "This is a standard system of equations.",
  "is_solvable": true,
  "model_correct": true,
  "payoff_correct": 10,
  "payoff_wrong": -5,
  "payoff_decline": 2,
  "utility_earned": 10
}
```

### 7.4 Kaggle Benchmarks SDK Mapping

Each of the six tasks becomes a `@kbench.task` function. The key architectural question is how to structure them.

**Option A (Recommended): One notebook, multiple tasks using .evaluate() with a DataFrame.**

Each task function takes the LLM and problem parameters, runs the full D1-D2-D3 pipeline for that problem, computes the relevant metric, and returns a numeric score.

The challenge: tasks are independent in the SDK, but our scoring requires aggregating across multiple problems (e.g., AUROC needs all bet-fractions and outcomes). 

**Solution:** Each task runs ALL problems internally and returns the aggregate metric as a float score. This means each task is self-contained but runs the full pipeline independently. This duplicates API calls across tasks.

**Better solution:** Run the full pipeline ONCE in a helper function, cache results, and have each task read from the cache. This requires careful state management but saves significant quota.

**Implementation approach:**
1. Define a global results cache (a dictionary stored in module scope)
2. The first task that runs executes the full pipeline and populates the cache
3. Subsequent tasks read from the cache and compute their specific metric
4. Use a lock or flag to prevent re-execution

**SDK-specific considerations:**
- `kbench.task` functions receive `llm` as a parameter — this is the model being evaluated
- Multi-turn conversations use `llm.chat()` or the conversation management API
- Structured output can use pydantic models or dataclasses
- `kbench.assertions.assess_response_with_judge` is used for the counterfactual plausibility scoring
- Tasks can return `float` for continuous scores (the SDK supports this via return type scoring)

### 7.5 Multi-Turn Conversation Implementation

The SDK supports multi-turn conversations. The critical implementation:

1. Create a chat session for each problem
2. Send D1 prompt → get response → parse
3. In the same session, send D2 prompt → get response → parse
4. In the same session, send D3 prompt → get response → parse
5. Close the session

The model retains context across all three turns, which is essential — D3 asks the model to reflect on its D1 predictions and D2 performance within the same conversational context.

If the SDK doesn't support persistent chat sessions, fall back to building message history manually and sending the full conversation each time.

### 7.6 Error Handling and Robustness

Things that will go wrong and how to handle them:

**Parsing failures:** Model doesn't follow the expected format.
- Implement fallback regex patterns (lenient matching)
- If a field can't be extracted, flag the problem as "unparseable" and exclude from that metric
- Report the parse failure rate — if it's >20%, the prompt needs revision

**All-correct or all-incorrect problems:** AUROC and Spearman are undefined.
- For AUROC: return 0.5 (chance) and flag
- For Spearman: exclude from average and report exclusion count
- Ensure problem difficulty is calibrated so this doesn't happen too often

**Model refuses to assess confidence:** Some models may hedge ("I can't know my confidence").
- The system prompt explicitly instructs honesty
- If the model still refuses, assign default bet_fraction = 0.5 (maximum uncertainty)
- Track refusal rate as a secondary metric

**API rate limits / quota exhaustion:**
- Implement progressive fallback: if quota is tight, reduce from 10 to 8 to 5 problems per level
- Prioritize completing at least one full pipeline over maximizing sample size
- Log all API calls with timestamps for quota tracking

---

## 8. Problem Generator Specification

### 8.1 L1 Problem Generator

The generator produces systems of linear equations with known solutions.

**Input parameters:**
- `num_variables`: 3 (fixed for 5-step problems)
- `coefficient_range`: tuple of (min, max) for random integer coefficients
- `solution_range`: tuple of (min, max) for the target solution values
- `seed`: random seed for reproducibility

**Generation process:**
1. Choose target solution values (e.g., x=2, y=-1, z=3)
2. Generate coefficient matrix A with random integers in range
3. Compute constant vector b = A × solution_vector
4. Verify the system has a unique solution (det(A) ≠ 0)
5. Determine the optimal solving strategy (which variable to eliminate first)
6. Compute each intermediate step's expected answer
7. Package into the problem record format

**Difficulty control:**
- Easy: coefficients in [-3, 3], solutions in [-5, 5], no fractions in intermediates
- Medium: coefficients in [-5, 7], solutions in [-10, 10], some fractional intermediates
- Hard: coefficients in [-10, 10], solutions in [-20, 20], frequent fractions, large intermediate values

### 8.2 L2 Problem Generator

Same as L1 but with operator replacement:

**Custom operator definitions:**
```
Operator Set Alpha:
  addition: a ⊕ b = a + b + 1
  subtraction: a ⊖ b = a - b - 1  
  multiplication: a ⊗ b = a × b + a + b
  (This makes ⊗ equivalent to (a+1)(b+1) - 1, which is associative)
  
Operator Set Beta:
  addition: a ⊕ b = 2a + b
  (This is NOT commutative, which creates interesting step-order dependencies)
  
Operator Set Gamma:
  All operations performed in modular arithmetic mod 13
  Division replaced by modular inverse
```

**Each L2 problem includes:**
1. The operator definitions as an in-context preamble
2. The problem using the novel operators
3. Ground truth computed using the novel operators

The structural template (which variable to eliminate, which step comes first) is identical to the corresponding L1 problem. Only the arithmetic changes.

### 8.3 Pre-Generation vs. Runtime Generation

**Decision: Pre-generate all problems and store as JSON.**

Reasons:
- Reproducibility — same problems across all models
- Verification — all ground truth answers checked before use
- Speed — no generation overhead during benchmark execution
- Debugging — problems can be inspected and reviewed

Generate 15 L1 problems (10 main + 5 feedback) and 15 L2 problems, at three difficulty levels (5 each). Store in JSON files attached to the Kaggle notebook as datasets.

---

## 9. Statistical Considerations

### 9.1 Sample Size and Power

With 10 problems per novelty level per task, our statistical power is limited. Here's what we can and can't detect:

**AUROC (Task 1):** 10 problems give us 10 (confidence, outcome) pairs. We can detect AUROC significantly different from 0.5 only for large effects (AUROC > ~0.8). This is acceptable for a benchmark — we're measuring the metric, not testing a null hypothesis.

**Spearman ρ (Task 2):** 10 problems × 5 steps = 50 (confidence, correctness) pairs if pooled, but correlations are computed per-problem and averaged. Individual problem-level ρ with N=5 needs very strong effects. The averaged ρ across 10 problems is more stable.

**Coherence (Task 4):** Location consistency is binary per problem, so 10 problems give us a proportion ± ~15% (binomial SE). Confidence consistency follows the same logic as Task 2.

**Metacognitive Control (Task 5):** 5 decision problems per novelty level. The utility-based scoring normalizes the model's total payoff to [0, 1] relative to worst and optimal (oracle) strategies. With only 5 decisions the score is coarse-grained (each decision shifts the score by ~0.1-0.3 depending on payoff magnitudes), but the accept/decline binary is robust to parse noise.

### 9.2 Reporting

For each score, report:
- Point estimate
- 95% bootstrap confidence interval (1000 resamples)
- Number of valid observations (after excluding unparseable responses)
- Parse failure rate

### 9.3 Effect Size Expectations

Based on the literature:
- **AUROC:** Frontier models typically show AUROC of 0.6-0.75 for verbalized confidence (Cacioli 2026). We expect similar.
- **Coherence:** This is genuinely unknown — no prior work measures prospective-retrospective coherence. Our hypotheses suggest it will be low (0.3-0.5 range).
- **Metacognitive Control:** We expect models to show reasonable accept/decline discrimination on L1 problems (utility ratio 0.5-0.8) but degraded performance on L2 (novel operators), especially for hard/high-risk problems and unsolvable problems.

---

## 10. Hypotheses and Expected Results

### Hypothesis 1: RLHF Incoherence
**Claim:** Instruction-tuned models will show moderate AUROC (0.6-0.7) but low prospective-retrospective coherence (<0.4).

**Rationale:** RLHF optimizes pre-answer responses and post-answer responses with different reward signals. Prospective hedging ("this might be hard") is trained by RLHF to produce appropriate epistemic humility. Retrospective explanation is trained to produce plausible post-hoc narratives. These two training signals don't enforce consistency between them.

**What would confirm it:** Coherence sub-score B (confidence consistency) below 0.4, while AUROC is above 0.6.

**What would falsify it:** High coherence alongside high AUROC, suggesting prospective and retrospective reports draw on the same internal signal.

### Hypothesis 2: Reasoning Models Show Best Step-Level Tracking
**Claim:** Models with chain-of-thought (o1-style, Gemini Thinking, DeepSeek-R1) will show the highest Spearman ρ (Task 2).

**Rationale:** Chain-of-thought explicitly externalizes step-by-step reasoning, which may create genuine per-step uncertainty signals. Standard models compress multi-step reasoning into a single forward pass, losing step-level uncertainty information.

**What would confirm it:** Reasoning model Spearman ρ > 0.3, while standard model ρ < 0.15.

**What would falsify it:** No significant difference between model types.

### Hypothesis 3: Reasoning Models Show Better Accept/Decline Calibration
**Claim:** Reasoning models will show higher metacognitive control scores (Task 5) than standard models, particularly on hard and unsolvable problems.

**Rationale:** Chain-of-thought reasoning externalizes the problem-solving process, which may give the model better access to genuine uncertainty signals. A model that can "see" itself struggling during reasoning should produce more accurate accept/decline decisions than one that compresses reasoning into a single forward pass.

**What would confirm it:** Reasoning model T5 utility ratio > 0.7, while standard model T5 < 0.6, with the gap widening on high-risk and unsolvable problems.

**What would falsify it:** No significant difference between model types, or standard models outperform reasoning models on accept/decline decisions.

### Hypothesis 4: Dissociated Coherence Sub-Scores
**Claim:** Models will show high location consistency but low confidence consistency.

**Rationale:** Matching a step number ("Step 3 was hardest") is a simple pattern-matching task. But maintaining consistent confidence narratives across time requires genuine metacognitive tracking, which LLMs likely lack.

**What would confirm it:** Location consistency > 0.6 while confidence consistency < 0.4.

**What would falsify it:** Both sub-scores correlated (either both high or both low).

---

## 11. Writeup Structure (1,500 words max)

The Kaggle writeup must follow their template. Here is the planned content allocation:

| Section | Target Words | Content |
|---|---|---|
| Project Name + Team | 20 | PRISM v2.1: Prospective-Retrospective Introspective Self-Model |
| Problem Statement | 200 | What's missing from current metacognition benchmarks. Why coherence matters. |
| Task & Benchmark Construction | 400 | The three-dimension architecture. Six tasks. How they map to cognitive science. |
| Dataset | 150 | Problem generation. L1/L2 design. Verification via stdlib math. |
| Technical Details | 300 | Scoring math. AUROC, Spearman, coherence composite. Statistical considerations. |
| Results, Insights, Conclusions | 350 | Model comparison. Hypothesis results. What this reveals about LLM metacognition. |
| References & Citations | 80 | Key papers (8-10 citations) |

**Total: ~1,500 words**

Key rhetorical moves for the writeup:
1. Open with the novel contribution: "No existing benchmark tests whether models' pre-task and post-task metacognitive reports are internally consistent."
2. Ground in cognitive science immediately: Nelson & Narens framework, DeepMind taxonomy alignment.
3. Emphasize discriminatory power: "The six tasks produce a gradient — models score differently on each dimension, revealing a metacognitive profile."
4. Present at least one surprising finding: Ideally Hypothesis 1 or 3 confirms.

---

## 12. Key Citations

| Citation | Role in PRISM |
|---|---|
| Nelson & Narens (1990) | Foundational metacognition framework (monitoring/control). Justifies the prospective-retrospective structure. |
| Maniscalco & Lau (2012) | Type-2 SDT methodology. Justifies AUROC as metacognitive sensitivity measure. |
| Fleming (2017) | Hierarchical Bayesian meta-d'. Cited for context, noting we use AUROC instead due to API constraints. |
| Guo et al. (2017) | ECE and modern calibration. Cited for context, noting we use AUROC for statistical reasons. |
| Cacioli (2026), arxiv 2603.25112 | M-ratio for LLMs. Key prior work. We cite the discretization limitation of verbalized confidence. |
| Wang et al. (AAAI 2025) | DMC framework. Prior art on decoupling metacognition from cognition. |
| DeepMind (2026) | Measuring Progress Toward AGI: A Cognitive Taxonomy. The competition's theoretical foundation. |
| Ackerman et al. (2025) | Evidence for Limited Metacognition in LLMs. Behavioral paradigms. Prior art on what LLMs can/can't do metacognitively. |
| Dai (2026), arxiv 2603.09309 | Rescaling confidence. Documents discretization artifacts in verbalized confidence. Justifies our bet-framing. |
| Chollet et al. (Jan 2026) | ARC Prize Technical Report. Contamination concerns. Justifies our L1/L2 novelty gradient. |

---

## 13. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Models don't follow structured format | High | Parsing failures reduce sample size | Lenient parsing, fallback patterns, exclude unparseable |
| All problems too easy (all correct) | Medium | AUROC/Spearman undefined | Include medium/hard difficulty problems, test in advance |
| All problems too hard (all incorrect) | Low | Same issue | Difficulty calibration with test runs |
| Quota exhaustion before completion | Medium | Incomplete results | Progressive reduction plan, prioritize core tasks |
| LLM-as-judge unreliable for counterfactuals | Medium | Coherence sub-score C noisy | Weight it at only 0.20, report raw scores alongside composite |
| Accept/decline decisions are trivial | Medium | Task 5 uninformative | Mix of risk profiles + unsolvable problems ensures non-trivial decisions |
| Novelty problems are too different | Medium | L2 performance drops for wrong reasons | Verify L2 problems match L1 structural difficulty exactly |
| Writeup exceeds 1,500 words | High | Possible penalty | Ruthless editing pass, cut Technical Details if needed |


---

## 15. Open Design Decisions

These decisions should be finalized during implementation:

1. **Step count per problem:** Currently set at 5. Could increase to 6-7 for better Spearman power, but longer problems increase API cost and parsing complexity.

2. **Number of decision problems per level:** Currently 5 (3 solvable + 2 unsolvable). Could increase for finer-grained utility scoring, but 5 is sufficient for a coarse signal while keeping API costs low.

3. **Counterfactual scoring weight:** Reduced to 0.20 in the coherence composite (from original 0.3), with location consistency at 0.35 and confidence consistency at 0.45. This reflects that LLM-as-judge scoring is noisier than the deterministic sub-scores.

4. **Task bundling strategy:** Run all 6 tasks independently (simple but expensive) vs. shared pipeline with caching (complex but efficient). The caching approach saves ~60% API calls but adds implementation complexity.

5. **Which task is the "main" task for the leaderboard:** The SDK requires one main task via `%choose`. The primary score is a weighted composite: `0.40 × mean(T4_L1, T4_L2) + 0.30 × mean(T5_L1, T5_L2) + 0.30 × min(T6, 1.0)`. This combines coherence (most novel construct), metacognitive control (accept/decline decisions), and novelty robustness.

6. **Structured output vs. free-text parsing:** The SDK supports pydantic models for structured output. If available for all test models, use it — eliminates parsing errors. If not universally available, use free-text with regex parsing.
