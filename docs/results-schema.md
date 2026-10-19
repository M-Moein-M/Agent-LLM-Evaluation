# Results schema

Everything a run leaves behind, and the aggregated files built from it. If you change a
field, update this file and AGENT.md in the same change.

## Trace layout

```
traces/<problem>/<agent>/<model-tag>/<campaign>/round-<N>/
```

- `<model-tag>`: the wire model id with `/` replaced by `_`, or `EVAL_MODEL_TAG` when set.
- `<campaign>`: `EVAL_DATE_TAG`, default `YYYYMMDD-HHMMSS` of the run start.
- One `host.sh` invocation writes one campaign per (model, problem) with rounds `1..EVAL_ROUNDS`.

## Files in a round directory

| File | Written by | Content |
|---|---|---|
| `agent.jsonl` (pi) | solve.sh | pi events: `message_end`, `tool_execution_start/end`, `agent_end`. Streaming deltas are dropped. |
| `agent.traj.json` (swe-mini) | solve.sh | mini trajectory: `messages[]`, each assistant message with `extra.response.usage`. |
| `agent.stream.jsonl` (claude, kimi) | solve.sh | claude: stream-json events. kimi: flat chat messages (`role`, `content`, `tool_calls`). Same name, different schema: tell them apart by content. |
| `agent.kimi-wire/main.wire.jsonl` (kimi) | solve.sh | kimi session wire log: `step.end` with per-call `usage` and `finishReason`, `content.part` with `think` parts. |
| `agent.stderr.log` | solve.sh | agent stderr (parser errors, retries). |
| `solution.json` | the agent | `{"build": [...], "run": [...], "variant_name": "..."}`. |
| `changed_diff.txt`, sources | host.sh | files the agent changed against the starter. |
| `.eval/facts.json` | measure.sh | what happened when the harness replayed the solution (below). No judgment. |
| `.eval/run.log`, `build.log`, `gpu_usage.csv`, `perf*.log`, `kern_*.csv` | measure.sh | raw evidence the checker reads. |
| `.eval/pass_result.json` | check_pass.py | the verdict (below). |
| `.eval/stats.json` | trace_stats.py | tokens, turns, end reason (below). |
| `.solve_timeout`, `.measure_timeout` | host.sh | markers when the watchdog or timeout killed a phase. |

## `.eval/facts.json`

`model`, `agent`, `round`, `date`, `size`, `variant`, `node`, `source` (`solution.json` or fallback),
`build` and `run`: `{cmd, status, start, end}` (epoch seconds, exit status).

## `.eval/pass_result.json`

```
{"passed": bool, "stages": [{"name", "passed", "value", "detail", "fatal", "skipped"}]}
```

Stages run in the order `checks/<problem>/check.py` returns them. `passed` is true when every
non-skipped stage passed. A stage named `metric` is non-gating; its `value` is
`{"metric", "value", "unit", "higher_better"}` and is what the problem ranks on.

## `.eval/stats.json`

| Field | Meaning |
|---|---|
| `agent`, `source` | parsed format and the trace file it came from |
| `turns` | model calls |
| `steps` | tool executions |
| `total_input`, `total_output`, `total_tokens` | summed over every model call (the prompt is re-sent each call). `null` when the agent logs no usage. claude uses the final `result` event totals because the gateway returns zero per-message usage. |
| `peak_context` | largest per-call context |
| `end_reason`, `end_severity`, `end_stop_reason`, `end_detail` | why the session stopped, in the vocabulary of `checks/failure_class.py` (`completed`, `context_overflow`, `max_turns`, `output_limit`, `cut_off`, ...). `end_severity` is `ok`, `warn` or `error`. |
| `capped_calls`, `capped_ctx` | calls that hit the server output cap, with the context size at each |
| `compacted`, `compaction_points` | context compaction events (agents are configured to avoid compaction) |
| `empty_reply_nudges` | times the agent had to be prompted after an empty reply |

## Aggregates (`tools/collect_results.py`)

`results/runs.jsonl`: one row per round: `problem`, `gpu`, `campaign`, `agent`, `model`, `model_raw`,
`round`, `passed`, `stages`, the `stats.json` fields above, `est_*` token estimates for old kimi
traces without a wire log, `fail_class` and `fail_detail` (first failing stage in the shared
vocabulary). `results/summary.json`: counts per failure class, agent and problem.

`tools/summarize.py` prints pass@1 and median tokens per (problem, agent, model) straight from the
`.eval` files without these aggregates.
