# Endpoint test

Checks that an endpoint works with all four agents, and what that costs.

```bash
tests/endpoint/run.sh --endpoint NAME --model KEY                 # prompts before spending tokens
tests/endpoint/run.sh --endpoint NAME --model KEY --yes --agents pi,swe-mini --steps 8
tests/endpoint/run.sh --endpoint NAME --model KEY --l0-only       # probes only, about 1K in / 2.5K out
```

| Part | What | Needs |
|---|---|---|
| L0, `probe.py` | 9 protocol probes: plain and streamed completion, tool call plain / streamed / parallel, reasoning field, `/v1/messages` plain / streamed tool use / thinking block | the endpoint |
| L1, `run.sh` + `check_trace.py` | one `tiled-gemm` round per agent, capped at `--steps` (default 8), then 8 checks on the trace: trace written, stats parsed, model calls, tool calls executed, usage recorded, end reason not an error, no parser errors, thinking present when `reasoning: true` | built `agent.sif` per agent (missing ones are skipped) |

L1 runs on a login node with `EVAL_GPU=0`, so the measure phase fails by design and is ignored. The test
asks whether the plumbing works, not whether the agent solves the problem. Traces go to
`traces/tiled-gemm/<agent>/endpoint-test/<stamp>/round-1`, logs to `eval-logs/endpoint-test/<stamp>/`, and
the report to `tests/endpoint/last-run.md`.

Status meaning: FAIL means an agent cannot work against this endpoint (for example tool calls leaking into
text, no tool call executed, no trace). WARN means it works with a limitation (no usage returned, no
thinking returned for a model declared as reasoning, one tool call per reply). Fixes for the common
FAILs are in `docs/gotchas.md` (G2, G4, G5, G6).

## Measured cost

Qwen3.6-35B-A3B-FP8 (reasoning model) behind LiteLLM, 8 steps, 2026-10-19, all 9 L0 probes and all
checks of all four agents PASS:

| Part | Tokens in | Tokens out |
|---|---|---|
| L0 probes | 968 | 2,547 |
| claude | 7,391 | 991 |
| kimi | 237,766 | 53,063 |
| pi | 253,597 | 43,343 |
| swe-mini | 22,191 | 596 |
| **Total** | **521,913** | **100,540** |

Input is summed over every model call. kimi and pi count the cached prefix each call, claude's total
comes from the final `result` event, so the per-agent numbers are not comparable with each other. Runs
of the same model varied by about 2x (claude, 8 turns: 7K to 30K in and 1K to 17K out across runs). A
model that thinks less costs less. Plan for up to about 1M in and 250K out as a worst case.
