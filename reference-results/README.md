# Reference results

The original author's evaluation results and the code that produced the figures. **This whole
folder is optional.** Delete it and the harness, tests, serving stack and tools keep working.

| Path | Content |
|---|---|
| `figures/` | example pictures. Give one to your coding agent and ask it to rebuild it from your own `results/runs.jsonl`. |
| `summary/` | `runs.jsonl` (one row per round), `summary.json`, `tables.md`, `notes.yaml`. Schema: `docs/results-schema.md`. |
| `analysis/` | reference scripts for tables, an HTML report and the figures. See `analysis/README.md`. |
| `traces-manifest.csv` | one row per campaign: problem, agent, model, rounds, pass counts, size, release asset name. |
| `trace-example/` | three rounds of one `tiled-gemm` campaign per agent (about 19 MB), same layout as `traces/`. Browse: `tools/trace-vis/run.sh -w reference-results/trace-example --scan-fs`. |
| `traces/` | full traces, gitignored. Extracted from the release assets below. |

## Fetching the full traces

The traces are GitHub release assets, one `traces-<problem>.tar.zst` per problem (about 0.7 GB in
total). They contain, per round, the agent trace, `solution.json`, `changed_diff.txt` and the small
`.eval/` evidence files. Workspace leftovers (binaries, volumes, profiler output) are not included.

```bash
gh release download <tag> --repo <owner>/<repo> --dir /tmp/assets
(cd /tmp/assets && sha256sum -c traces-assets.sha256)
for f in /tmp/assets/traces-*.tar.zst; do zstd -dc "$f" | tar -x -C reference-results; done
tools/trace-vis/run.sh -w reference-results --runs summary/runs.jsonl     # browse them
```

## Producing the assets (maintainers)

`python3 reference-results/build_trace_assets.py --traces traces --out DIR --rules PRIVATE_RULES.tsv`
packs an allow-list of files and applies regex rules to names and contents (to strip account and host
names). Keep the rules file outside the repo.

## Reproducing the numbers

```bash
python3 tools/collect_results.py --traces reference-results/traces --out /tmp/runs.jsonl --summary /tmp/summary.json
python3 reference-results/analysis/report_tables.py --runs reference-results/summary/runs.jsonl
EVAL_RUNS_JSONL=reference-results/summary/runs.jsonl EVAL_FIGURES_OUT=/tmp/figs \
  python3 reference-results/analysis/make_figures.py
```

The figures need matplotlib and numpy. The tables and figures apply the author's exclusions
(`EXCLUDE_CAMPAIGNS` in `analysis/report_tables.py`) and an A40-only filter.
