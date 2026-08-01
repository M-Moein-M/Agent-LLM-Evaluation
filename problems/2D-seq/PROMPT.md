Make this solver run as fast as possible on this machine.

`src/solver.c` contains a 2D Poisson solver: successive over-relaxation on a
uniform grid, driven by `poisson.par`. It is currently sequential. It is the
only part of the program that matters for runtime.

Build and run it, work out where the time goes, and make it faster. How you do
that is up to you.

Rules:

- Do not edit `poisson.par`. The harness re-runs the solver with this exact
  file.
- Do not edit `src/verify.c` or `src/verify.h`. They print the `DIGEST` line
  the harness grades against.
- Do not edit `src/main.c`. It brackets `solve()` with the timer and prints the
  `SOLVE_TIME` line you are scored on.
- The `DIGEST` line your build produces must match the reference to within a
  relative tolerance of `1e-9` on every field. The reference comes from the
  unmodified solver on the same input, so this is a statement about the
  numerics you are allowed to change — not about floating-point luck.
- The reported time is the `SOLVE_TIME` line, which covers `solve()` only.
  Setup time is not counted, but it is also not free — the harness measures
  total wall time as well.

Everything else is fair game: the build system, compiler flags, the data
layout, `src/solver.c`, `src/allocate.c`, and any new files you add.

When you are finished, write `solution.json` in the problem root with the exact
commands you used:

```json
{
  "build": ["make"],
  "run": ["./exe-GCC poisson.par"],
  "variant_name": "..."
}
```

Requirements:
- All commands must be relative to the problem root directory.
- `variant_name` is a short label for the approach you took.
- These commands will be independently re-run by the evaluation harness for
  verification. If your solution does not build, or its `DIGEST` does not
  match, the harness will fail it — write commands that actually work.
