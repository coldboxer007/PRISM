# PRISM v2.1 — Prospective-Retrospective Introspective Self-Model

A metacognition benchmark for the Kaggle "Measuring Progress Toward AGI"
Hackathon (Cognitive Abilities — Metacognition Track).

## What It Measures

Whether LLMs can accurately monitor their own reasoning **before** (D1),
**during** (D2), and **after** (D3) solving multi-step math problems.

## Six Tasks

| # | Task | Metric |
|---|------|--------|
| 1 | Prospective Calibration | AUROC |
| 2 | Step-Level Accuracy | Spearman ρ |
| 3 | Retrospective Self-Assessment | Accuracy |
| 4 | **Coherence** (primary) | Weighted composite |
| 5 | Metacognitive Control | Accept/decline utility |
| 6 | Novelty Robustness | L2/L1 ratio |

## Structure

- `problems/` — Deterministic problem generator (Types A/B/C × L1/L2)
- `prompts/` — Prompt templates for D1, D2, D3, and decision problems
- `scoring/` — Parsers, step scorer, and metrics engine
- `tasks/` — One module per task; each exports a `compute_task_N()` function
- `pipeline.py` — Orchestrator: runs D1→D2→D3 per problem, caches results
- `notebook.py` — Kaggle Benchmarks notebook entry point
- `validate.py` — Offline validation suite (`python -m prism_v2.validate`)
