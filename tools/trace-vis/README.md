# trace-vis

A small local web UI for reading agent trajectories out of `traces/`. Browse by
problem, by agent or by model, drill down to a run, and read its trajectory as a vertical
timeline with markdown rendering and collapsible tool input/output.

Ships as an Apptainer image so starting it is one command.

## Quick start

```bash
cd tools/trace-vis
./build.sh          # -> trace-vis.sif  (needs network for the base image only)
./run.sh            # binds the repo read-only, prints the URL. Needs results/runs.jsonl:
                    # python3 ../collect_results.py --traces ../../traces --out ../../results/runs.jsonl
```

Then open the printed `http://localhost:<port>`. On the cluster, forward it:

```bash
ssh -N -L 8080:localhost:8080 you@login-node
```

`run.sh` options: `-p PORT`, `-w WORKSPACE`, `-s SIF`, `--open` (bind `0.0.0.0`
instead of localhost), `--scan-fs`. Anything else is passed through to `app.py`, e.g.
`--traces DIR` and `--runs FILE` (relative to the workspace). To browse the reference traces:
`./run.sh -w ../../reference-results --runs summary/runs.jsonl`. The default scope is every run;
`TRACE_VIS_SCOPE=curated` selects the author's curated subset.

Without the wrapper:

```bash
apptainer run -B /path/to/repo:/workspace:ro trace-vis.sif --port 8137
```

The workspace is bound **read-only**. This is a viewer; it never writes to
`traces/`.

### Without a container

Stdlib-only, no dependencies, works on any Python 3.8+:

```bash
python3 app.py --workspace /path/to/repo --port 8080
```

## Navigation

The three pivots reach the same runs from different directions:

- **By problem** — problem → agents that attempted it → campaigns → runs
- **By agent** — agent → campaigns → problems → runs
- **By model** — model → agents that drove it → campaigns → problems → runs

The agent view pivots on the campaign because one submission runs a model
against every problem under a single timestamp. That wave is the unit you
compare, so it is the heading — model + timestamp, newest first — with the
problems it covered listed under it, separated by a rule, each with its own
rounds table. The problem view keeps the other order.

The model view has one level more than the other two: a section per agent, and
inside each section the agent view's layout. Holding the weights fixed and
reading down the agents is how you tell a model's ceiling apart from a harness
that fails to reach it. Agent headings link across to the full agent view.

Round links open in a new tab. A rounds table is a list you work through, and
a timeline is a long read; coming back with the browser's Back button would
lose your place in the list.

### Which runs the lists show

The header switch picks the scope, and it sticks:

- **Important runs** (default, 932 of 1393) — exactly the set behind
  `reference-results/figures/05_pass_matrix.png`: A40 rounds only, latest campaign per
  problem × agent × model, campaigns under 3 rounds dropped, known
  harness/config bugs excluded. Superseded campaigns and smoke tests stay out
  of the way, and every cell of that figure reproduces from this scope.
- **All runs** — every row in `results/runs.jsonl`, including the CPU-only
  2D-seq study, which the pass matrix's A40 filter leaves out. Rounds outside
  the curated set are dimmed in the tables.

The rules live in `curation.py`, ported from `reference-results/analysis/make_figures.py`
(`load_latest_all()`) with `EXCLUDE_CAMPAIGNS` from `reference-results/analysis/report_tables.py`;
`selftest.py --workspace` fails if that exclusion set drifts from the
workspace's. The merges (claude-pro → claude, `gemma-4-31B-it` →
`…-FP8-block`, `…-0731-think` → `…-0731`, `Kimi-K2.7-Code-NVFP4` →
`Kimi-K2.7-Code`) decide *selection* only — runs still appear under the agent
and model they actually ran as, so navigation keeps matching `traces/`.

A run is always openable by link, whatever the scope; a run outside it says so
at the top.

A run page shows the pass/fail state, the metric, token and turn totals, the
check stages, the round's side files (`solution.json`, `changed_diff.txt`,
`build.log`, …, loaded on demand), and the timeline itself. Filter chips toggle
event kinds; long traces load 400 events at a time.

### Tool inputs and outputs

Agents hand tool results back as JSON, so a shell call arrives as
`{"returncode": 0, "output": "/problem\ntotal 16\n…"}` — printed raw, every line
break shows up as the two characters `\n`. The viewer decodes the JSON instead:
scalar fields collapse into one `key: value` line and each multi-line field
becomes its own labelled block, printed as real text and copyable verbatim.

Only text that actually decoded as JSON is unescaped. A payload that is already
plain text is shown byte for byte — `\n` inside source code, a diff or a log
line is two characters there, not a line break, and rewriting it would misreport
what the agent saw.

### Why a run stopped (pi, kimi, claude, swe-mini)

A pi trace that outgrew the model's context window just stops mid-conversation —
the reason lives in the trailing `agent_end` envelope, inside a provider error
string two JSON levels deep, and nothing in the timeline showed it. The parser
now closes every pi trace with a `run_end` event: the reason in words, the two
token counts when the provider gave them, named for what they actually count
(`220,211 input tokens against a 215,684 limit — 4,527 over`; a provider that
reports input+completion says `tokens in total` instead), a bar for how much of the request fit, and the raw
provider message behind a disclosure.

Reasons: `completed`, `context_overflow`, `output_limit`, `rate_limit`,
`timeout`, `api_error`, `tool_use`, `cut_off` (the run ends after a tool result
with no final assistant message — turn limit, walltime, or the harness stopping
it). Across the 476 pi rounds here: 315 completed, 68 output limit, 41 context
overflow, 23 cut off, 16 timeout, 13 rate limit.

kimi's own trace file (`agent.stream.jsonl`) never records why a run stopped
either way — a harness-side cutoff (per-turn step cap, a dropped connection, a
provider error) is only ever written to the sibling `agent.stderr.log`, as the
CLI's own terminal line (`error: failed to run prompt: <domain>.<kind>: …`). The
parser reads that line for the closing `run_end` event, reusing pi's reason
vocabulary where the underlying provider message matches, plus three kimi-only
reasons: `max_steps_exceeded` (the turn hit its configured step cap),
`connection_error` (the connection to the provider dropped), and
`empty_response` (the model returned no text or tool call to continue on).
Across the 309 kimi rounds here: 218 completed, 43 max steps exceeded, 14 empty
response, 11 api error, 9 context overflow, 8 connection error, 2 output limit,
2 rate limit, 1 cut off — plus one round where both the trace and its stderr
log are empty, with no signal left to classify.

claude and claude-pro do write the ending down, in the single `result` record
that closes `agent.stream.jsonl` — but reading it takes knowing the vocabulary.
The one that matters is the turn budget: a run the harness cut at
`EVAL_MAX_TURNS` ends exactly like a run that finished — last tool call
answered, nothing after it — and the two differ only in that record
(`subtype: "error_max_turns"`, `is_error: true`, `errors: ["Reached maximum
number of turns (120)"]` versus `subtype: "success"`, `is_error: false`). One of
them left a half-finished edit on disk; the timeline could not tell you which.
The parser classifies from `terminal_reason` (`completed` / `api_error` /
`max_turns` / `blocking_limit`), then from the provider text in `result`, adding
two claude-only reasons: `max_turns` and `thinking_block` (the gateway rejecting
an assistant thinking block — a known LiteLLM streaming bug). A killed process
writes no `result` record at all, and that absence is itself the signal:
`cut_off`. Across the 448 claude/claude-pro rounds here: 311 completed, 50 turn
limit, 33 rate limited, 26 context overflow, 11 thinking block, 6 cut off, 5
timeout, 3 api error, 2 connection error, 1 output limit.

swe-mini writes its ending as a terminal `role:"exit"` message whose
`extra.exit_status` is a bare class name. It used to render as a log line
reading `session exit: ContextWindowExceededError` — the words were there, but
not what they cost: the run in `RabbitCT-LolaBunny/swe-mini/Qwen3.8-27B` had
sent 258,274 tokens into a 256,000-token window, and the number was buried in
`extra.exception_str`. The parser now closes the trajectory with the same
`run_end` node as the other three, mapping `Submitted` to `completed`,
`LimitsExceeded` to `limits_exceeded` (naming the cap that was hit, from
`info.config.agent.step_limit` / `cost_limit` against `info.model_stats`) and
`RepeatedFormatError` to `format_error` (the model produced
`max_consecutive_format_errors` malformed tool calls in a row) — but
`RepeatedFormatError` is only the symptom, so the parser reads the interrupt
records behind it before believing it. A model whose reply was cut off
mid-tool-call (`finish_reason: "length"`) sends no tool call either, and that is
not a formatting problem: in
`RabbitCT-LolaBunny/swe-mini/Qwen3.8-27B/20260818-051152/round-3` all three
final replies were truncated after ~570 tokens on top of a 255,430-token
prompt, i.e. the 256,000-token window was full and there was no room left to
answer in. Those runs are reported as `context_overflow` (or `output_limit`
when the prompt was small and the reply itself hit a cap), with a note saying
what filled the window. Genuine format errors — the model answered, it just
never called a tool — keep `format_error`. Anything else
is classified from the exception text, so litellm's `ContextWindowExceededError`
yields both token counts and the bar. A trajectory with no exit message was
killed before it could write one: `cut_off`. Across the 433 swe-mini rounds
here: 314 submitted, 73 format error, 27 limits exceeded, 10 context overflow, 8
cut off, 1 api error.

### Why a round failed

Every failing round carries ONE fine-grained reason, `fail_class`, shown on the
fail chip (`fail: context_overflow`) and in the run header's `why` row, with the
supporting evidence — the compiler line, the exception, the token counts — in
the tooltip. It replaces what that chip used to say, the name of the first
failed stage, which was the word `build` for nearly half of all failures and
answered nothing: those rounds mostly never reached a compiler, the agent had
died first.

The classifier is `checks/failure_class.py`. It lives in the eval pipeline, not
here, and is copied into the image by `%files`, so the viewer and
`results/summary.json` can never disagree about why a round failed. It reads
only receipts already on disk (`.eval/facts.json`, `build.log`, `run.log`, the
`run_end` reason above) and never re-runs anything. Precedence:

1. **A failed graded stage wins** (`mse`, `gpu`, `perf`, `digest`, `integrity`).
   The code built and ran; there is a measurement, and the measurement is the
   reason.
2. **Otherwise the harness's artifact finding** — `no_solution`,
   `incomplete_solution`, `make_config_stop`, `toolchain_missing`,
   `make_no_rule`, `link_error`, `ptxas_error`, `compile_error`, `missing_file`,
   `build_timeout`, `run_timeout`, `run_error`. `run.log` is quoted at its first
   error line, not its last: a runner prints its usage banner *after* the thing
   that went wrong.
3. **But when the finding is "there is no artifact" and the session also ended
   badly, the `run_end` reason wins.** A missing `solution.json` is the symptom;
   `context_overflow` / `format_error` / `max_turns` is the cause.

Across the 1214 failing rounds here: 286 no solution written, 274 MSE, 105
context overflow, 77 output limit, 73 format error, 63 run error, 46 turn limit,
40 max steps, 38 make stopped after config.mk, 36 cut off, 27 perf, 22 limits
exceeded, 21 timeout, 17 compile error, 14 empty response, 14 GPU, 12 api error,
and a tail of link/toolchain/ptxas errors. Not one reads `build`.

**Pass/fail arithmetic is untouched.** `passed`, `stages` and `failed_stage` are
still written exactly as before; anything that is not a pass still counts as a
failure, and every pass@1 is the number it always was (597 / 1811 here, byte for
byte the same as before this change).

## How it works

```
app.py          stdlib HTTP server: JSON API + static files
indexer.py      builds the browse index
curation.py     the default "important runs" scope (pass-matrix selection)
parsers.py      normalizes the 4 trajectory schemas into one event model
failure_class.py  shared with checks/ — why a round failed (copied in at build)
build_web.py    the frontend "build": bundles + hashes the JS/CSS
selftest.py     offline checks, used as the image build gate
jstest.js       payload-rendering checks, run under node via `selftest.py --js`
static/         source frontend (vanilla JS, no bundler, vendored libs)
```

**The index is seeded from `results/runs.jsonl`, not from a directory walk.**
`traces/` is ~19 GB and ~1450 round dirs on a cluster filesystem; `runs.jsonl`
has one row per measured round with everything navigation needs, plus `__path`
pointing at the round. Index build is ~0.04 s. `--scan-fs` additionally walks
the tree to surface rounds that never reached the checker — it is opt-in and
never on the request path.

So a campaign that is on disk but **not yet in `runs.jsonl` reads as `unknown
run`**. Regenerate the file (`python3 tools/collect_results.py`) — the server
notices the new mtime and reindexes by itself — or start with `--scan-fs` to
list rounds straight from the tree, without metrics.

Parsed traces are cached by `(path, mtime)`. Per-field text is capped at 20 000
characters and the timeline at 4000 events, with the UI marking anything it
truncated. The largest trace here is 75 MB / 293 events and pages in ~1.3 s
cold, ~0.01 s warm.

### API

| Endpoint | Purpose |
| --- | --- |
| `/api/summary` | totals, problems, agents, models |
| `/api/problem?name=` | the problem-first pivot |
| `/api/agent?name=` | the agent-first pivot |
| `/api/model?name=` | the model-first pivot (agents nested inside) |
| `&scope=curated\|all` | on the four above; default `curated` |
| `/api/run?run=<run_id>` | run metadata, schema, available side files |
| `/api/trace?run=&offset=&limit=&kinds=` | a page of normalized events |
| `/api/file?run=&name=` | one whitelisted side file |
| `/api/reindex` | rebuild the index |

`run_id` is `problem/agent/model/campaign/round-N`. It is a query parameter, not
a path segment, because it contains slashes.

Only the files in `ALLOWED_FILES` (`app.py`) can be fetched, capped at 512 KB.
Round paths are resolved through a traversal guard that confines them to
`traces/`. All model-generated markdown goes through DOMPurify before it reaches
the DOM.

## The four trajectory schemas

| Agent | File | Shape |
| --- | --- | --- |
| `pi` | `agent.jsonl` | line-delimited envelopes (`session`, `message_end`, …) with `toolCall` / `toolResult` blocks |
| `swe-mini` | `agent.traj.json` | one JSON doc, OpenAI-style `messages` list, `exit` role at the end |
| `claude`, `claude-pro` | `agent.stream.jsonl` | line-delimited Anthropic `/v1/messages` stream records (`type: system\|assistant\|user\|result`) |
| `kimi` | `agent.stream.jsonl` | line-delimited **raw** OpenAI chat messages — no envelope `type` field at all |

`claude` and `kimi` share a filename, so `detect_kind()` peeks at the first
line: a top-level `role` key means kimi (even its `role: meta` records carry
one), its absence means claude. If the file is empty — kimi failing before it
writes any output at all, e.g. rate limited on the first request — it falls
back to `agent.stderr.log`: a `error: failed to run prompt:` terminal line is
the kimi CLI's own format, never produced by the other harnesses.

All four normalize to one event list. Kinds: `system`, `user`, `assistant_text`,
`thinking`, `tool_call`, `log`, `run_end`. Tool results are stitched back onto the call
that produced them via the call id, so a `tool_call` event carries both its
input and its output; a result whose call is missing becomes a `log` event
rather than vanishing.

### Adding a fifth schema

1. Add the filename to `detect_kind()` in `parsers.py` (and the disambiguation,
   if it collides with an existing filename).
2. Write `_parse_<name>(path)` returning `{"events", "session_meta", "warnings"}`,
   building events with `_new_event()` and stitching tool results through a
   `pending_calls` dict keyed by call id.
3. Dispatch to it in `parse_trace()`.
4. Add a fixture to `selftest.py` (`fixture_*` + the `FIXTURES` list) — the
   shared assertions then apply to it automatically.

If you introduce a new event *kind*, add it to `KINDS_PY` in `selftest.py`,
`FILTER_KINDS`/`LABEL` in `static/js/timeline.js`, and a `.ev.<kind>::before`
rule in the CSS. The selftest fails if those three drift apart — otherwise an
unknown kind would be filtered out of the timeline with no visible sign.

## Frontend build

There is no npm. `build_web.py` is the whole toolchain: it concatenates the
vendored libs (marked, DOMPurify) and our modules in load order, hashes the
result, writes `bundle.<hash>.js` / `style.<hash>.css`, and rewrites
`index.html` to point at them. The Apptainer `%post` runs it, then runs
`selftest.py` as a gate, then re-verifies the built tree — so a parser
regression, a missing asset or unbalanced JS fails the build instead of shipping
a blank page.

The server picks its frontend from `--static` / `$TRACE_VIS_STATIC`, defaulting
to the source tree. That means editing `static/` and reloading works with no
rebuild; the image sets the variable to the built bundle.

## Checks

```bash
python3 selftest.py                                   # offline: schemas, assets, JS lint
python3 selftest.py --workspace /path/to/workspace    # also parses real traces
python3 selftest.py --js --workspace /path/to/ws      # also runs the JS under node
apptainer run --app selftest trace-vis.sif --workspace /workspace
```

Offline they cover the parsers, the curation rules, the assets and a JS lint;
`--workspace` adds real traces, the curated/all scope split, and the
`EXCLUDE_CAMPAIGNS` cross-check against `checks/report_tables.py`.

`--js` runs `jstest.js`, which loads `timeline.js` and asserts on the payload
decoding — against synthetic cases and, with `--workspace`, against events
pulled from real traces. It needs a JS runtime: `node` on `PATH`,
`$TRACE_VIS_NODE`, or else `apptainer exec docker://node:22-slim` (network on
the first pull). Deliberately **not** a build gate: there is no node in the
image, and the build must stay offline.

The offline checks synthesize a minimal round dir for each of the four schemas,
so they run during the image build with nothing bound. Since there is no node in
the image, the JS check is a bracket/quote balance scan — a lint, not a parser,
but it catches the realistic failure of a stray brace blanking the page. It
self-tests against known-good and known-bad snippets so a broken linter cannot
pass everything.
