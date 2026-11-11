# Analysis scripts

Reference implementations, written for the author's data. Use them as a starting point and let your
coding agent adapt them to yours.

| Script | Does |
|---|---|
| `report_tables.py` | markdown ranking tables per problem (pass@1 next to the problem's metric). `--runs`, `--gpu`. |
| `report_html.py` | the same as a sortable standalone HTML report, joins `summary/notes.yaml`. Default output `results/report.html`. |
| `make_figures.py` | the main figure set from `results/runs.jsonl` (`EVAL_RUNS_JSONL` overrides the input, `EVAL_FIGURES_OUT` the output directory, default `results/figures`). |
| `make_figures_stats.py` | statistical companions: confidence intervals, retry gain, cost per solve. Imports `make_figures`. |
| `make_figures_examples.py`, `make_figures_coinflip.py` | cross-problem examples, a coin-flip noise illustration. |
| `kimi_token_estimate.py` | backfills token estimates for old kimi traces that have no wire log (see below). |
| `_extract_kimi_prefix.py`, `_mock_openai_server.py` | captured kimi's constant injected prompt prefix from a stub server, input to the estimate. |

Aggregation itself (`traces/` to `runs.jsonl`) is `tools/collect_results.py`, which stays in the
harness so that `tools/trace-vis` works without this folder.

Author-specific choices live in code, not in config: `EXCLUDE_CAMPAIGNS`, `PROBLEM_ORDER`,
`A40_ERA_START` in `report_tables.py`, the model alias and display-name tables in `make_figures.py`,
the price table in `make_figures_stats.py`, `TOKENIZER_MAP` in `kimi_token_estimate.py`. Replace them
for your own models.

Install: `pip install matplotlib numpy pyyaml`. `kimi_token_estimate.py` also needs `transformers`
and a `tokenizers/` directory next to it (gitignored) with one Hugging Face tokenizer per model,
named like the trace model folder (`Qwen_Qwen3.6-27B`, ...). Download the tokenizer files of each
model into it, e.g. with `huggingface-cli download <repo> --include "tokenizer*" --local-dir tokenizers/<name>`.
