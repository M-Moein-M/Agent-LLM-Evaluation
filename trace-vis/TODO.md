# trace-vis — TODO

A minimal local web UI for browsing and reading agent trajectories from
`traces/`, shipped as a single Apptainer image you start with one command.

**Status: done and verified end to end.** See `README.md` for usage. What
follows is the original plan with outcomes, kept for the constraints it records
and the ideas not pursued.

---

## Context / constraints discovered

- Trace layout: `traces/<problem>/<agent>/<model>/<timestamp>/round-<N>/`
  — 5 problems, 5 agents, 14 models, ~1448 round dirs, **19 GB** total.
- **Do not walk `traces/` to build the index.** `results/runs.jsonl` already has
  1393 rows with `problem, agent, model, model_raw, campaign, round, passed,
  stages, gpu, turns, steps, total_input/output/tokens, peak_context, compacted`
  and `__path` (= `<round>/.eval/pass_result.json`). One file read, instant.
  Walking the tree is a *fallback* for rounds with no eval result, and must be
  opt-in / lazy — never on the request path.
- **4 different trajectory schemas** (this was the main parsing work):

  | agent | file | shape |
  |---|---|---|
  | `pi` | `agent.jsonl` | JSONL envelope events: `session`, `message_start/_end`, `tool_execution_start/_end`. Assistant content blocks `text` / `thinking` / `toolCall`; results arrive as a separate `role:"toolResult"` message keyed by `toolCallId`. |
  | `swe-mini` | `agent.traj.json` | ONE JSON doc: `{info, messages[], trajectory_format}`. Plain OpenAI chat: `assistant.tool_calls[]` + `role:"tool"` keyed by `tool_call_id`. Has a terminal `role:"exit"`. |
  | `claude`, `claude-pro` | `agent.stream.jsonl` | JSONL with `type`: `system`(init) / `assistant` / `user` / `result`. Anthropic blocks `text` / `thinking` / `tool_use`; results are `tool_result` blocks inside a `user` message keyed by `tool_use_id`. |
  | `kimi` | `agent.stream.jsonl` | JSONL of **raw OpenAI messages, no `type` field**. `role` ∈ assistant/tool/meta. `meta` rows carry rate-limit retries + resume hints. |

  `claude` and `kimi` share a filename → disambiguate by peeking whether the
  first line has a `type` key.
- Some traces are **huge** (largest `agent.stream.jsonl` = 75 MB). Per-field text
  caps are not enough — need an event cap + pagination.
- No network at run time inside the container → `marked` + `DOMPurify` are
  vendored into `static/vendor/`.
- Flask is not installed on the login nodes → the server is **stdlib only**, so
  it can be smoke-tested on the host and the image build can't break on a pip
  mirror.

---

## Tasks

### 1. Parsing — `parsers.py`
- [x] Normalize all 4 schemas to one event shape:
      `{kind, role, text, tool_name, tool_call_id, tool_input, tool_output, is_error, timestamp}`
      with `kind ∈ system | user | assistant_text | thinking | tool_call | log`.
- [x] Stitch tool results back onto their originating call by id; emit an
      orphan `log` event when the id is unknown.
- [x] Cap per-field text at `MAX_TEXT` and flag `*_truncated`.
- [x] Event cap + offset/limit pagination (`slice_events`) so a 75 MB trace
      can't wedge the browser; payload carries `total_events` / `has_more`.
- [x] `selftest.py` parses one trace per schema and asserts tool calls got their
      outputs attached — against synthetic fixtures (so it runs during `%post`
      with nothing bound) and, with `--workspace`, against real traces.
- [ ] Extract per-run token/usage totals from the *trace* (pi `message.usage`,
      claude `result`). Not needed in the end: `runs.jsonl` already carries
      `turns/steps/total_tokens/peak_context`, and the run header uses those.

### 2b. Default scope — `curation.py` (added after the first pass)
- [x] Port `slides/make_figures.py::load_latest_all()` + `EXCLUDE_CAMPAIGNS`:
      A40 only, ≥ 3 rounds, latest campaign per problem × agent × model,
      identity merges (claude-pro→claude, gemma bare tag→FP8-block,
      0731-think→0731, NVFP4→Kimi-K2.7-Code), per-round artifact drops.
- [x] Verified: all 33 cells of `slides/figures/05_pass_matrix.png` reproduce
      from the curated scope — 932 of 1393 runs.
- [x] `scope=curated|all` on `/api/summary|problem|agent`, curated by default;
      `/api/run` reports whether the run is in it, and stays openable either way.
- [x] Header switch, persisted in `localStorage`; a note under each heading says
      what is hidden and links to the other scope; excluded rounds are dimmed in
      the "all" tables.
- [x] Ported rather than imported: `make_figures.py` needs matplotlib and the
      image builds with nothing bound. `selftest.py --workspace` diffs
      `EXCLUDE_CAMPAIGNS` against `checks/report_tables.py` so they can't drift.
- [ ] Not done: the merges are selection-only. The UI still lists `claude-pro`
      as its own agent and the gemma/NVFP4 tags as their own models, because
      those are the directories the traces live in.

### 2. Index — `indexer.py`
- [x] Load `results/runs.jsonl` → in-memory index; derive `round_dir` from
      `__path`. Group into problem → agent → model → campaign(timestamp) → rounds.
- [x] Build the two navigation pivots: `by_problem[problem][agent]` and
      `by_agent[agent][problem]`.
- [x] Stable `run_id` (`problem/agent/model/campaign/round-N`) + `round_path()`
      with a guard that confines the result to `traces/`.
- [x] Fallback filesystem scan for rounds absent from `runs.jsonl` — opt-in via
      `--scan-fs` only.
- [x] `maybe_refresh()` rebuilds on an mtime change of `runs.jsonl` (a cheap
      stat, no tree walk); `/api/reindex` forces it.

### 3. Server — `app.py` (stdlib `ThreadingHTTPServer`, not Flask)
- [x] JSON API — `/api/summary`, `/api/problem`, `/api/agent`, `/api/run`,
      `/api/trace`, `/api/file`, `/api/reindex`. `run_id` is a **query param**,
      not a path segment, because it contains slashes.
- [x] Whitelisted side files only (`ALLOWED_FILES`), capped at 512 KB.
- [x] Parsed-trace LRU keyed by `(path, mtime)`.
- [x] Serve the SPA shell + static, with unknown non-API paths falling back to
      `index.html` so deep links and refresh work.
- [x] `--workspace`, `--host`, `--port`, `--scan-fs`, `--static` + env fallbacks.

### 4. UI — `static/`
- [x] SPA shell + hash routing, no bundler.
- [x] **Home**: totals, per-problem and per-agent cards, entry into both pivots.
- [x] **By problem**: problems → agents → campaigns → runs.
- [x] **By agent**: agents → problems → campaigns → runs.
- [x] Run list rows: round, pass/fail chip, metric, turns, steps, tokens.
- [x] **Run view = vertical timeline**: connector rail, one node per event,
      colour per kind, tool calls showing `tool_name` + a one-line input
      summary, `<details>` for full input/output, markdown for user/assistant
      text, monospace `<pre>` for tool output, errors flagged red, filter chips,
      run header with problem · agent · model · round · verdict · schema.
- [x] Decode tool payloads instead of dumping them (added after the first pass):
      a JSON result like `{"returncode": 0, "output": "…\n…"}` printed its line
      breaks as the characters `\n`. `payloadBlocks()` decodes the JSON, keeps
      scalars on one line and gives each multi-line field its own labelled
      block. Plain text is left byte for byte — unescaping it would rewrite
      `\n` inside source code and diffs. JSON cut short by `MAX_TEXT` still gets
      its escapes decoded, since it is known to be JSON.
- [x] Collapsible side panels: `solution.json`, check stages, and every other
      whitelisted file, fetched on first expand.
- [x] `style.css`: one accent, system fonts, dark default + light toggle.

### 5. Packaging
- [x] `trace-vis.def` — `python:3.12-slim` base, no pip installs. `%post`
      **builds the frontend** (`build_web.py`), then runs `selftest.py` as a
      build gate, then re-verifies the built bundle. `%test` repeats both;
      `--app selftest` re-runs them inside the finished image.
- [x] `build_web.py` — the whole frontend toolchain: concatenate vendored libs +
      modules in load order, hash, emit `bundle.<hash>.js` / `style.<hash>.css`,
      rewrite `index.html`.
- [x] `build.sh` — host selftest first, then `apptainer build --fakeroot`.
- [x] `run.sh` — binds the workspace read-only, picks a free port, prints the
      URL and the `ssh -L` line.
- [x] `README.md` — build, run, port-forward, API, and how to add a 5th schema.

### 6. Verify
- [x] `selftest.py` green: 108 offline checks, 126 with `--workspace`.
- [x] The JS is now actually executed, not only balance-linted: `jstest.js`
      loads `timeline.js` and asserts on the payload decoding (32 checks),
      run by `selftest.py --js`. No node on the login nodes and none in the
      image, so the runner falls back to `apptainer exec docker://node:22-slim`
      and the whole thing stays opt-in, never a build gate.
- [x] One run opened per agent (`pi`, `swe-mini`, `claude`, `claude-pro`,
      `kimi`) — every tool call has its output attached, zero orphans.
- [x] The 75 MB `tiled-gemm/claude/.../round-10` trace: 1.29 s cold parse,
      293 events, 646 KB for a 200-event page, 0.01 s for the next page.
- [x] Both navigation pivots reach the same run (checked through the container).
- [x] Image built (43 MB) and served end to end from `run.sh`.

---

## Notes / decisions
- The workspace is bound **read-only**; the UI never writes to `traces/`.
- Port default `8080`, `run.sh` walks up to 8130 for a free one; cluster access
  via `ssh -L`.
- The event-kind vocabulary is duplicated in `parsers.py`, `timeline.js` and the
  CSS. `selftest.py` asserts the three agree — otherwise a new kind is silently
  filtered out of the timeline (this actually happened with `system`).

## Possible next steps (not requested)
- Compare view: two runs side by side, or a campaign's rounds diffed.
- Full-text search across a run's events, or across all traces.
- Link a run to its `perf.log` / `stats.json` as a small chart.
