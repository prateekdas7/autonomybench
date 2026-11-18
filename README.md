# AutonomyBench

A benchmark for whether LLM agents choose the right level of autonomy when acting for a user. Built for CS 6983 (Group 2: Prateek Das, Raisa Vikas Furtado, Hanish Mehla).

## What it measures

Each episode is a two stage decision:

1. **Stage 1:** Decline, Clarify, or Proceed
2. **Stage 2** (only if Proceed): Recommend, Prepare, Request Approval, or Execute

Scenarios come in matched A/B pairs. Both versions share the same instruction and tools, and only one factor in the world changes: ambiguity, authorization, magnitude, reversibility, or third party impact. The agent has to find that factor with its read tools. A model that responds to the factor should change its decision or level between A and B.

Agents act in a mock world (inbox, calendar, files, shop, payments, policies) through 15 plain Python tools. Nothing real is sent or paid. Every tool call is logged, and a classifier reads the log to assign the decision and level.

## Repo layout

```
autonomybench/      harness code (schema, world, tools, graph, classifier, judge, scorer)
configs/            experiment settings (models, sampling, limits)
prompts/            agent system prompt and judge prompt
scenarios/pilot/    pilot scenario pairs (JSON)
scenarios/full/     full benchmark scenario pairs
templates/          scenario templates used to generate pairs
labels/             team gold labels for agreement checks
results/runs/       episode trajectories (jsonl, not committed)
results/summaries/  aggregated results
scripts/            run, score, summarize, validate
tests/              unit tests
```

## Local setup (plumbing tests only)

1. Install [Ollama](https://ollama.com) and pull a small model that fits a laptop GPU:
   ```
   ollama pull qwen3.5:4b
   ```
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Check the scenarios:
   ```
   python scripts/validate_scenarios.py pilot
   ```

The small local model is only for checking that the harness works. Its decisions are not results.

## Running experiments (Kaggle)

Real runs use free Kaggle GPUs (T4 x2, about 30 GPU hours per week per account).

1. Verify your phone number in Kaggle settings, so notebooks can use the internet.
2. Create a notebook with **Accelerator: GPU T4 x2** and **Internet: On**.
3. Run the setup cell from `00_smoke_test.ipynb`. It installs Ollama, starts the server with one model loaded at a time and an 8192 token context, pulls the models, and installs packages.
4. In the setup cell, pull only the agent model you are running plus `phi4` (the judge).
5. Stop the session when done. Kaggle counts wall clock time, so idle sessions use up quota.

Agent models: `qwen3.5:9b`, `gemma4:12b`, `gpt-oss:20b`. Judge: `phi4`.

## Adding scenarios

Put each scenario as `scenarios/<split>/<scenario_id>.json`, for example `P01_A.json` and `P01_B.json`. Run the validator before committing. It checks required fields, valid gold labels, that the evidence exists in the world, and that A and B differ only in the world.