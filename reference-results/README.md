# Reference results

The original author's evaluation results and the code that produced the figures. **This whole
folder is optional.** Delete it and the harness, tests, serving stack and tools keep working.

| Path | Content |
|---|---|
| `figures/` | example pictures. Give one to your coding agent and ask it to rebuild it from your own `results/runs.jsonl`. |
| `summary/` | `runs.jsonl` (one row per round), `summary.json`, `tables.md`, `notes.yaml`. Schema: `docs/results-schema.md`. |
| `analysis/` | reference scripts for tables, an HTML report and the figures. See `analysis/README.md`. |
| `trace-example/` | one sample round per agent. See its README. |

## Full traces

The 1,053 sessions behind the pass@1 matrix (`figures/05_pass_matrix.png`) are on Hugging Face:
[m-moein-m/Agent-LLM-Evaluation-Traces](https://huggingface.co/datasets/m-moein-m/Agent-LLM-Evaluation-Traces).
Each row is one round with the full message history, tool calls, reasoning where recorded, and the
checker result. Account names, hosts and addresses are redacted. Rounds outside that population (older
campaigns, superseded reruns) appear only as metrics in `summary/runs.jsonl`.

## Reproducing the numbers

```bash
python3 reference-results/analysis/report_tables.py --runs reference-results/summary/runs.jsonl
EVAL_RUNS_JSONL=reference-results/summary/runs.jsonl EVAL_FIGURES_OUT=/tmp/figs \
  python3 reference-results/analysis/make_figures.py
```

The figures need matplotlib and numpy. The tables and figures apply the author's exclusions
(`EXCLUDE_CAMPAIGNS` in `analysis/report_tables.py`) and an A40-only filter.
