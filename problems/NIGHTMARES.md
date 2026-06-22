# HPC Mini-Nightmares — problem catalog

> ⚠️ **Authoring doc — contains spoilers.** This file lives at `problems/` root,
> *not* inside any `problems/<name>/` directory, so the harness never copies it
> into an agent workspace (`host.sh` copies only the single problem dir). Keep
> bug answers here, never inside a problem's own files.

Inspired by debug-gym's
[mini_nightmare](https://github.com/microsoft/debug-gym/blob/main/data/mini_nightmare/mini_nightmare.md),
adapted to the [Berkeley HPC dwarfs](https://patterns.eecs.berkeley.edu/) and the
parallel/numerical bugs this project cares about (see AGENT.md → *HPC Dwarfs*).

## What makes an HPC mini-nightmare

Per AGENT.md, a good entry is:

- **Semantically correct, contextually wrong** — compiles, runs, often produces
  *almost*-correct output.
- **No syntax errors / obvious smells** — the buggy line looks idiomatic.
- **Requires a mental model outside the code** — thread scheduling, a physics or
  numerical convention, an MPI protocol detail, a launch-config invariant.
- **Bug and symptom are far apart** — a launch-config truncation shows up as a
  band of wrong voxels; a missing barrier shows up as a tiny error at large K.
- **One-line fix, long reasoning.**

### The lever this harness gives you

The harness replays the **agent's own** `build`/`run` from `solution.json` — it
does *not* inject a hidden test at a different scale. So a robust nightmare must
make the *mandated* verification expose the bug. Two patterns:

1. **In-binary fixed verifier** (used by `tiled-gemm`): ship a "do-not-modify"
   verifier that sweeps sizes / thread counts / message sizes and computes a CPU
   reference + MSE inside the binary. The PROMPT mandates the exact run command;
   the bug surfaces in the printed `Mean Squared Error`. Tampering with the
   verifier is visible in the captured `changed_diff.txt`.
2. **New check stages** (not yet built): a **non-determinism** stage (run N
   times, flag output variance — for races) and a **scaling/timing** stage (for
   "correct but no speedup" bugs like false sharing). These need small additions
   to `checks/eval_lib.py`.

---

## Status legend
✅ implemented · 🟡 designed, not built · 💡 idea

---

## ✅ tiled-gemm — Dense linear algebra
**Bug:** the tiled CUDA GEMM kernel is fully correct *and* bounds-checked for
partial tiles, but the launch truncates the grid:
`dim3 grid(N / TILE, N / TILE)` instead of `(N + TILE - 1) / TILE`. When `N` is
not a multiple of `TILE` (16), the trailing rows/cols of `C` are never computed
and read back as 0.
**Why it's a nightmare:** the kernel body *advertises* partial-tile handling
(`row < N && aCol < N` guards, `(N+TILE-1)/TILE` tile loop), so a reader concludes
remainders are handled — but the launch never creates those blocks. Symptom (a
band of zeros) is far from the cause (one integer-division line). Invisible at
the "natural" demo size 256; the verifier sweeps `{128, 200, 256, 257, 384}`.
**Fix:** `dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);`
**Check:** `build → run → mse(≤1e-2) → gpu`. Files: `problems/tiled-gemm/`,
`checks/tiled-gemm/check.py`.

---

## ✅ stream-compact — Dynamic / data-dependent parallelism (CUDA)
**Task (generate):** implement order-preserving stream compaction — keep every
element `> threshold`, preserve input order, return the count. No Thrust/CUB.
**Bug it traps:** the obvious parallel approach, `pos = atomicAdd(counter, 1);
out[pos] = x;`, produces the right *set* of elements in a **nondeterministic
order** — the output position is a *scan dependency* (it depends on how many
kept elements precede `x`), which a racing atomic counter cannot honor.
**Why it's a nightmare:** breaks the embarrassingly-parallel map — there is no
correct per-element-independent write; you need a prefix sum + scatter. The
failure is a runtime race: "right elements, wrong order, sometimes." Compiles,
runs, looks plausible on a single eyeball check.
**Validation (the point):** the fixed verifier runs the agent's `compact()`
**32×** on one random permutation, compares each result to an order-preserving
CPU reference element-by-element, and reports the **worst-case** error as a
single `Mean Squared Error` line — so the nondeterminism collapses into the
existing MSE check with **no new harness machinery**. Correct = exactly 0.
**Check:** `build → run → mse(≤1e-9) → gpu`. Files: `problems/stream-compact/`,
`checks/stream-compact/check.py`.

---

## Backlog

### 🟡 syncthreads-gemm — Dense linear algebra (CUDA)
Shared-memory tiled GEMM with the `__syncthreads()` after the tile *load*
dropped. Compiles, "mostly works"; tiny errors that grow with K / certain block
dims, and vary run-to-run. **Needs the non-determinism check stage.**
Fix: restore the barrier. Canonical CUDA race.

### 🟡 buffer-parity — Structured grids (CPU/OpenMP)
Jacobi/double-buffered stencil swaps `old`/`new` pointers but returns `old`;
correct for even iteration counts, off-by-one-sweep for odd. Reference uses an
even count, verifier uses odd. Fix: swap the return / fix the parity. Pure MSE.

### 🟡 csc-spmv-atomic — Sparse linear algebra (OpenMP)
Column-parallel SpMV writes `y[row]` from multiple threads without `atomic`.
Single-thread correct; multi-thread drops contributions. A diagonally-dominant
smoke test hides it. **Needs non-determinism + thread-count sweep.**
Fix: `#pragma omp atomic` on the update (or row-parallel CSR).

### 🟡 false-sharing — Reduction (OpenMP, performance-only)
`long counter[NTHREADS]` packed on one cache line. **Correct result, ~zero
speedup.** Only a scaling/roofline metric can catch it — the one nightmare that
*requires* this project's multi-core/-node advantage to even detect.
**Needs the scaling/timing check stage.** Fix: pad to cache-line stride.

### 💡 isend-reuse — MPI
Non-blocking `MPI_Isend`/`Irecv` reuses the send buffer before `MPI_Wait`. Works
for small messages (eager protocol), corrupts for large ones — *size-dependent*,
the archetypal "bug far from symptom." Identical on 1 rank. Needs MPI in the
container. Fix: `MPI_Wait` before reuse.

### 💡 reduce-vs-allreduce — MPI
Iterative solver uses `MPI_Reduce` (only rank 0 gets the global residual) where
`MPI_Allreduce` was meant; other ranks use stale residuals and diverge.
Identical on 1 rank. Fix: `MPI_Allreduce`.

### 💡 fft-normalization — Spectral methods
Forward+inverse FFT missing the `1/N` scaling (or applying it twice). Right
shape, wrong magnitude — passes an eyeball check, fails MSE. Fix: normalize once.

### 💡 minimum-image — N-body / MD
Periodic-boundary minimum-image convention using `round()` vs the intended
`nearbyint`/`nint` sign convention, or cutoff compared against `r` while the rest
is squared. Subtly wrong neighbor lists; energy drifts slowly. Complements the
existing MD-Bench SIMD task.

---

## Adding a new nightmare (checklist)

1. `problems/<name>/` — source with the planted bug + `PROMPT.md` (mandate the
   exact build/run, say "fix the kernel, not the verifier", don't reveal the bug)
   + a fixed in-binary verifier that prints `Mean Squared Error: <x>`.
2. `checks/<name>/check.py` — `Check(ProblemCheck)` with ordered stages and
   threshold constants (clone `checks/tiled-gemm/check.py`).
3. Add the spoiler entry here.
4. Run: `EVAL_PROBLEMS="<name>" ./scripts/run_eval.sh` (or add to a comma list).
