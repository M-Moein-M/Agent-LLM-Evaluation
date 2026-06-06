# TODO — Evaluation Harness

## 1. agent.jsonl is too big

- **Status:** Resolved (2026-06-06) — `guest.sh` now filters the event stream before
  `tee`, dropping the streaming delta events `message_update` and
  `tool_execution_update`. On the real 272MB Qwen3.6/RabbitCT trace this cut the file
  to **1.4MB (99.5% reduction)**: 28,819 → 797 lines, retaining all 199 message_end,
  104 tool_execution_end, turn boundaries, and the full `agent_end` transcript.
- **Severity:** High
- **File:** [guest.sh:29](agents/docker/pi/guest.sh#L29)
- **Symptom:** One run (Qwen3.6-35B, RabbitCT) produced a ~200MB `agent.jsonl` file. Over 20 rounds × multiple models this will eat disk fast.
- **Context:** We switched from `agent_end`-only to full event stream (`tee ~/agent.jsonl`).
- **Root cause (measured):** `message_update` was **99.7%** of the file (282MB / 27,828
  events, avg 10KB each). It fires per streaming token and re-serializes the *entire*
  accumulated message each time → quadratic. Every byte is already captured losslessly
  in the matching `message_end` (and again in `agent_end`).
- **Decision:** Refined option 3 — filter out the `*_update` deltas with
  `grep -Ev '"type":"(message_update|tool_execution_update)"'` in the pipe. Keeps full
  tool-call granularity (unlike reverting to `agent_end` only) and never writes the
  wasted bytes at all (unlike gzipping the full stream). Nothing downstream reads
  `agent.jsonl` (judging uses `.eval/facts.json` + logs), so the cut is lossless for the
  harness. `grep` is wrapped in `{ ... || true; }` so an empty agent run still hits the
  fail-handling path under `set -euo pipefail`. If even tighter is wanted later, gzip can
  be layered on top.

## 2. nvidia-smi log resolution

- **Status:** Resolved (2026-06-04) — `guest.sh` now runs a single background `nvidia-smi --format=csv,nounits -lms 250 -f gpu_usage.csv` writer for the duration of the run, killed afterward. Utilization aggregation (avg/peak) lives in `checks/eval_lib.py:gpu_stats` and the threshold is applied in `checks/RabbitCT/check.py`.
- **Severity:** Medium
- **File:** [guest.sh:100-104](agents/docker/pi/guest.sh#L100-L104)
- **Current:** Manual loop with `sleep 1` (1-second resolution). 4 samples over a 50s run is too coarse.
- **User suggestion:** Use `nvidia-smi -lms 100` for 100ms auto-sampling, e.g.:
  ```
  nvidia-smi --query-gpu=timestamp,index,name,uuid,pci.bus_id,power.draw,clocks.sm,clocks.mem,utilization.gpu,temperature.gpu --format=csv,nounits -lms 100 -f <output_file>
  ```
- **User feedback:** "250ms would be good for now" — compromise between resolution and I/O overhead. `nvidia-smi` doesn't support 250ms directly, but `-lms 250` is also valid.
- **TODO:** Replace the manual `sleep 1` loop with `nvidia-smi -lms 250`. Run it as a background process during the harness's `run` phase, then stop it when the run finishes. This gives ~200 data points over a 50s run.
- **Pass criteria integration:** Compute average GPU utilization from these samples and apply a threshold (e.g., >20% = real GPU, ≤20% = likely CPU fallback).

## 3. Move pass checks and pass@1 to Python with per-problem config

- **Status:** Resolved (2026-06-04) — Implemented host-side. `guest.sh` now only captures raw evidence (`.eval/facts.json` + logs + `gpu_usage.csv`). Judgment moved to Python: `checks/eval_lib.py` (`ProblemCheck` base + accessors), per-problem `checks/<PROBLEM>/check.py` (subclass with ordered stages + threshold constants), `checks/check_pass.py` (host-side driver → `.eval/pass_result.json`), `checks/collect_results.py` (pass@1 + per-stage failures). The whole judging subsystem lives under `checks/` (engine + per-problem definitions together, outside the agent workspace so it can't be gamed); `scripts/` keeps only the bash entrypoint `run_eval.sh`. Re-runnable over existing traces with no GPU cost. Differs from the original proposal below: thresholds are constants in `check.py` (not an `eval.json`), and the verdict is a separate `pass_result.json` (not merged into the facts file).
- **Severity:** High
- **Files involved:**
  - [guest.sh:111-112](agents/docker/pi/guest.sh#L111-L112) — MSE/RMSE extraction (bash + awk)
  - [guest.sh:93-113](agents/docker/pi/guest.sh#L93-L113) — build/run/replay logic
  - [collect_results.py](scripts/collect_results.py) — pass@1 calculation (currently basic)
- **Problem:** Pass criteria are scattered across guest.sh (bash), run.log parsing, and collect_results.py. Hard to add per-problem thresholds.
- **Design goal:** Single place to define per-problem pass criteria, evaluated consistently.
- **Proposed structure — `problems/<name>/eval.json`:**
  ```json
  {
    "pass": {
      "build_must_succeed": true,
      "run_must_succeed": true,
      "mse_threshold": 0.01,
      "gpu_required": true,
      "gpu_min_avg_utilization": 20,
      "gpu_min_peak_utilization": 50
    }
  }
  ```
- **Implementation plan:**
  1. Create `eval.json` for RabbitCT with MSE threshold + GPU utilization checks
  2. Add per-problem config loading (guest.sh or a new eval step reads it)
  3. Centralize pass verdict in `run_result.json` as `{"pass": true/false, "checks": {"build": "pass", "run": "pass", "mse": "pass", "gpu": "fail"}}`
  4. `collect_results.py` reads the per-problem `eval.json` and applies thresholds to compute pass@1
  5. (Optional) Move entire evaluation logic from guest.sh to a Python script

- **Why Python?**
  - JSON parsing is first-class (vs `jq` + string manipulation in bash)
  - Statistical analysis (GPU utilization averages, percentiles, confidence intervals)
  - Threshold comparison with clear error messages ("gpu_avg_util: 12.3% < 20% threshold")
  - Easier to extend for new problem types
  - More maintainable than shell scripts with heredocs

## 4. MSE parsing bug — "0" becomes empty string

- **Status:** Resolved (2026-06-04) — The bash/awk extraction was removed from `guest.sh` entirely. MSE is now parsed host-side in `checks/eval_lib.py:log_value` via `re` + `float()`, so `"0"` correctly yields `0.0` (not `""`), and a missing/unparseable value yields `None`.
- **Severity:** Medium
- **File:** [guest.sh:111](agents/docker/pi/guest.sh#L111)
- **Bug:** The awk gsub regex `/[^0-9.eE+-]/` uses `.` inside a character class. In awk regex, `.` inside `[...]` is still a wildcard matching any character. So this regex effectively removes everything except digits, `e`, `E`, `+`, `-`. The intended behavior was to keep numeric characters and the literal decimal point, but `.` as a wildcard means the gsub is removing `0` and any other digits in certain contexts.
- **Impact:** When MSE is exactly `0` (perfect match), it becomes an empty string in `run_result.json`. This breaks pass@1 calculation (empty string can't be compared to a threshold).
- **Fix:** Use a stricter regex: `gsub(/[^0-9]/, "", $2)` to keep only digits, or handle the `.` case explicitly: `grep -oP '\d+\.?\d*' "$RUN_LOG" | head -n 1`
- **Seen in:** First eval run (Qwen3.6, RabbitCT)

---

## Previous decisions (for reference, not action items)

- **Agent writes code + `solution.json`, harness replays commands** — gives agent debugging agency (MT@4 in one pass), harness independently verifies
- **One output format for all problems** — `solution.json` works across RabbitCT, MD-Bench, etc. — only prompt changes per problem
- **`agent.jsonl` full event stream** — was implemented but disk usage is a concern (see TODO #1)
- **`nvidia-smi -lms` suggestion** — user recommended this over manual loop. Use `-lms 250` for 250ms resolution.
