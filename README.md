# AutonomyBench
CS 6983 Final Project, Group 2: Prateek Das, Raisa Vikas Furtado, Hanish Mehla
AutonomyBench measures whether LLM agents choose an appropriate level of autonomy.
Stage 1: Decline, Clarify, or Proceed. Stage 2 (if Proceed): Recommend, Prepare,
Request Approval, or Execute. Scenarios come in matched pairs that change one factor
(ambiguity, authorization, magnitude, reversibility, third party impact).
## Setup
1. Install Python 3.11+ and [Ollama](https://ollama.com).
2. Pull the models listed in `configs/conditions.yaml` (`ollama pull <tag>`).
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and adjust if needed.
## Usage
```
python scripts/generate.py --set pilot
python scripts/run.py --condition qwen --scenarios pilot --trials 1
python scripts/run.py --condition always_execute --scenarios pilot --trials 1
python scripts/score.py --runs results/runs
python scripts/summarize.py --scored results/scored.jsonl
python scripts/judge_agreement.py --labels labels/judge_check.csv
```
Conditions are defined in `configs/conditions.yaml`.
## Repo layout
| Path | Contents |
|---|---|
| `autonomybench/` | world, tools, graph, baselines, classifier, judge, scorer, metrics |
| `configs/` | conditions and judge settings |
| `prompts/` | system prompt and judge prompt (hashed into every run record) |
| `templates/` | one scenario template per factor |
| `scenarios/` | generated scenario JSON |
| `scripts/` | entry points |
| `results/` | run records (JSONL) and summaries |
| `labels/` | rubric labels and judge check labels |
| `tests/` | unit tests |