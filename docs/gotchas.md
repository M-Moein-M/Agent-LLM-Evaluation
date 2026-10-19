# Gotchas

Problems that cost time when found by trial and error. Each entry names the symptom, the cause,
and the fix. Grouped by where they bite.

## Endpoint and serving

**G1. The model server moves.** A self-hosted endpoint gets a new address whenever its job is
restarted. Update `base_url` in `endpoints.yaml` and run `tests/endpoint/run.sh --l0-only` before a
campaign. Stage 0 of `scripts/run_eval.sh` does the same one-token probe per model and aborts in
about 30 s instead of failing hours into a batch.

**G2. The server's tool-call parser must match the model.** Symptom: the agent does one turn and
stops, no `solution.json`. Tool calls leaked into `content` as text and the agent executed
nothing. Example: Qwen3.6 emits `<tool_call><function=NAME><parameter=ARG>...` and needs SGLang
`--tool-call-parser qwen3_coder`, not `qwen25`. Probe P3 and P4 of the endpoint test detect this.

**G3. Same wire id, new checkpoint.** A provider can swap the weights behind an unchanged model id.
Keep the wire id and set `trace_tag` (`EVAL_MODEL_TAG`) so the two checkpoints land in different
trace folders. Streaming usually reveals the true id.

**G4. Long generations die on default timeouts.** Three layers each need a limit above the longest
reply: LiteLLM `timeout`/`stream_timeout`, SGLang router `--request-timeout-secs` (default 1800
silently cuts streams at 30 minutes), and each agent's own client timeout (pi settings, kimi patch).
Keep LiteLLM `fake_stream: false`. With fake streaming nothing is sent until the reply is complete,
and a slow model trips kimi's 300 s timeout in a retry loop that lasts until the walltime limit.
See `serving/litellm/`.

**G5. The `claude` agent through LiteLLM needs three config keys.** `/v1/messages` over SGLang works
only with the settings shipped in `serving/litellm/config.template.yaml`. The hidden one is
`supports_reasoning: false` on the model entry. A streaming bug in LiteLLM 1.89.4 that dropped
thinking blocks and killed the claude agent on turn 1 for reasoning models is fixed by
`serving/litellm/patches/`. Earlier thinking is forwarded as `thinking_blocks`, not
`reasoning_content`, and the first chunk of each streamed block can be dropped.

**G6. Reasoning models and the `-think` alias.** Whether thinking is on is decided by the served
chat template. Some setups expose a second id with a `-think` suffix. Use `agent_ids` in
`endpoints.yaml` to give claude a different wire id than the other agents. Probes P6 and P9 report
whether reasoning comes back on each protocol.

**G7. Kimi K2.x intermittently omits `</think>`.** SGLang's reasoning parser then dumps the whole
generation into `reasoning_content` and the tool-call parser sees nothing: the reply is empty with
no tool calls. This is a model bug seen on SGLang, vLLM and Foundry. Dropping `--reasoning-parser`
removes the forced-empty reply. Clients differ in what they do with a degenerate turn (G9).

## Agents

**G8. Completion by absence vs by explicit signal.** claude, kimi and pi treat a turn with no tool
call as "done". A garbage turn therefore looks like a finished run: exit 0, empty stderr, no
`solution.json`, nothing edited. swe-mini never ends implicitly. A turn without a tool call is a
format error (three in a row end as `RepeatedFormatError`), and the run ends only when the model
executes `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` and the command returns 0. Do not compare pass
rates across agents without looking at `end_reason`, and do not read exit 0 plus empty stderr as an
attempt.

**G9. Kimi crash-loop on empty replies.** kimi's client requires text or tool calls and raises
`APIEmptyResponseError` otherwise. pi skips empty assistant turns when serializing and ends the
loop. claude through LiteLLM does not validate and ends on the turn budget. Newer Kimi Code
versions did not change this check (0.34.0 compared to 0.31.0).

**G10. kimi's step counter.** A step is one model call, the same unit as swe-mini `step_limit` and
pi `--step-limit`. Kimi logs `turnStep=<turn>/<step>` in `kimi-code.log` under its session dir.
Rounds can exceed the cap: some show more than 120 assistant lines although stderr reports
`Turn exceeded maxSteps=120`. Unresolved. The wire log (`agent.kimi-wire/`) is the reliable
source for per-call usage and finish reasons.

**G11. Claude Code blocks requests itself at 177K unless told otherwise.** It assumes a 200K window
for every non-Anthropic model and stops with `Prompt is too long` (`terminal_reason:
blocking_limit`) when `estimate >= window - min(max_output, 20000) - 3000`. With
`CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000` that is 177,000. The estimate is the last reported usage plus
characters/3 for newer messages, and a thinking-only response is counted twice, so the block can fire
well before the real input is that large. The server never sees these requests. `claude/solve.sh`
sets `CLAUDE_CODE_BLOCKING_LIMIT_OVERRIDE` to 10,000,000 (`EVAL_CLAUDE_BLOCKING_LIMIT`) so that the
server's own context error ends the round, like the other agents. Traces from before that change
(campaigns up to mid-September) contain rounds cut at the client limit.

**G12. At the server's output cap.** kimi always ends the round (it looks `completed` unless the
wire log is read: `output_limit`), pi ends on a text-only reply, claude and swe-mini continue.
Set the cap deliberately per model (`max_output_tokens`, `EVAL_MAX_OUTPUT_TOKENS`).

**G13. `claude -p` ends "successfully" when the assistant replies without a tool call.** Check the
raw `result` record in `agent.stream.jsonl`, not only the parsed events.

**G14. Agent processes escape the container.** Without a private PID namespace an agent `pkill`
kills sibling rounds on the same node. Every `host.sh` runs `apptainer exec --pid`.

**G15. Context handling is unified: no compaction.** pi has compaction disabled in its settings,
kimi is patched (`make_patched_kimi.sh`) to end the run on a context overflow instead of
compacting, claude runs with auto-compaction off. A run that fills the window ends as
`context_overflow`, the same way for every agent.

## Harness

**G16. Stage a job workdir with `rsync -a`, never `cp -a`.** `cp -a` tries to copy ACLs, which many
cluster filesystems reject (one error line per file). The measure phase also decides "did the agent
modify this file?" from size and mtime via `rsync --compare-dest`, so a copy that resets mtimes
flags every file as modified.

**G17. nsys lives in the image.** The perf grader prefers `nsys` kernel timing and falls back to
cudaEvent timing when it is absent. The install is non-fatal, so a failed install is easy to miss.
After building an image run `scripts/smoke_perf.sh` and confirm the verdict says `(nsys)`.

**G18. Measure needs a quiet GPU, solve does not.** Solve containers share a GPU (kernel time slices),
measure runs serially under a lock because timing is the graded quantity. Readings taken while
another job used the GPU can be a third of the true speed. `scripts/remeasure.sh` replays solutions
from traces.

**G19. Nothing is persisted until the end of a campaign.** `host.sh` writes traces and runs the
checks at the end. A walltime kill or a hang can lose the whole campaign, not just the stuck round.
Wrap campaigns in `timeout`, keep the walltime above the watchdog limits, and check `traces/` after
any interruption.

**G20. Timeouts.** `EVAL_SOLVE_STALL_TIMEOUT` kills a solve whose trace file has not changed for an
hour, `EVAL_SOLVE_TIMEOUT` is the 4 h backstop, `EVAL_MEASURE_TIMEOUT` caps the measure phase.
A killed round leaves `.solve_timeout` or `.measure_timeout` in its trace directory.

**G21. A bulk-input problem needs its data bound.** RabbitCT mounts `data/RabbitCT` read-only. Do not
copy bulk data into the workdir. The empty mountpoint left in the workdir is pruned from the trace.

**G22. `check_pass.py` and `trace_stats.py` are idempotent re-judges.** They rewrite
`.eval/pass_result.json` and `.eval/stats.json` for every round they find. Edit a threshold in
`checks/<problem>/check.py`, rerun with `--traces traces`, and verdicts are recomputed with no GPU.
`host.sh` limits them to its own campaign with `--campaign`.
