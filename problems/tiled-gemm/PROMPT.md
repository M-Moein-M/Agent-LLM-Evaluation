A tiled CUDA implementation of single-precision matrix multiplication
(`C = A * B`) lives in `src/`. It compiles and runs, but the built-in
verification reports a mismatch against the CPU reference for some matrix sizes.

Find and fix the defect so that `./gemm -c` reports a near-zero **Mean Squared
Error** for every size it tests.

Constraints:
- Keep the tiled GPU approach. Do **not** replace it with a CPU-only or naive
  implementation.
- Do **not** modify the verification harness in `src/main.cu` — the sizes, the
  reference, and the error metric there are fixed and are how you are graded.
- The fix is small. Build and run the verification yourself and iterate until it
  passes.

When finished, write a file `solution.json` in the problem root with the exact
commands you used:

```json
{
  "build": ["make"],
  "run": ["./gemm -c"],
  "variant_name": "tiled-gemm"
}
```

Requirements:
- All commands must be relative to the problem root directory.
- These commands will be independently re-run by the evaluation harness for
  verification.
- If your solution does not compile or still reports a non-trivial Mean Squared
  Error, the harness will fail — write commands that actually work.
