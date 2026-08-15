# trace-vis

A small local web UI for reading agent trajectories out of `traces/`. Browse by
problem or by agent, drill down to a run, and read its trajectory as a vertical
timeline with markdown rendering and collapsible tool input/output.

Ships as an Apptainer image so starting it is one command.

## Quick start

```bash
cd trace-vis
./build.sh          # -> trace-vis.sif  (needs network for the base image only)
./run.sh            # binds the repo read-only, prints the URL
```

Then open the printed `http://localhost:<port>`. On the cluster, forward it:

```bash
ssh -N -L 8080:localhost:8080 you@login-node
```

`run.sh` options: `-p PORT`, `-w WORKSPACE`, `-s SIF`, `--open` (bind `0.0.0.0`
instead of localhost), `--scan-fs`. Anything else is passed through to `app.py`.

Without the wrapper:

```bash
apptainer run -B /path/to/nhr-eval-workspace:/workspace:ro trace-vis.sif --port 8137
```

The workspace is bound **read-only**. This is a viewer; it never writes to
`traces/`.

### Without a container

Stdlib-only, no dependencies, works on any Python 3.8+:

```bash
python3 app.py --workspace /path/to/nhr-eval-workspace --port 8080
```

## Navigation

Both pivots reach the same runs from opposite directions:

- **By problem** — problem → agents that attempted it → campaigns → runs
- **By agent** — agent → problems it attempted → campaigns → runs

### Which runs the lists show

The header switch picks the scope, and it sticks:

- **Important runs** (default, 932 of 1393) — exactly the set behind
  `slides/figures/05_pass_matrix.png`: A40 rounds only, latest campaign per
  problem × agent × model, campaigns under 3 rounds dropped, known
  harness/config bugs excluded. Superseded campaigns and smoke tests stay out
  of the way, and every cell of that figure reproduces from this scope.
- **All runs** — every row in `results/runs.jsonl`, including the CPU-only
  2D-seq study, which the pass matrix's A40 filter leaves out. Rounds outside
  the curated set are dimmed in the tables.

The rules live in `curation.py`, ported from `slides/make_figures.py`
(`load_latest_all()`) with `EXCLUDE_CAMPAIGNS` from `checks/report_tables.py`;
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

## How it works

```
app.py          stdlib HTTP server: JSON API + static files
indexer.py      builds the browse index
curation.py     the default "important runs" scope (pass-matrix selection)
parsers.py      normalizes the 4 trajectory schemas into one event model
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
| `&scope=curated\|all` | on the three above; default `curated` |
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

`claude` and `kimi` share a filename, so `detect_kind()` peeks at the first line:
a `type` key means claude, its absence means kimi.

All four normalize to one event list. Kinds: `system`, `user`, `assistant_text`,
`thinking`, `tool_call`, `log`. Tool results are stitched back onto the call
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
