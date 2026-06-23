A tiled CUDA implementation of single-precision matrix multiplication
(`C = A * B`) lives in `src/`. It compiles and runs, but the built-in
verification currently reports a mismatch against the CPU reference for some
matrix sizes — and it is far from the throughput the GPU is capable of.

This is a **performance-optimization task**. Your goal is to make the GPU GEMM
as **fast as possible** while staying correct. `./gemm -c` verifies correctness
and prints its own throughput line — use it to iterate. Correctness is a hard
floor (a near-zero **Mean Squared Error** for every size it tests).

**How you are graded:** the evaluation does *not* trust any number your program
prints. It links your kernel into its own benchmark, runs your kernel **and
cuBLAS** on one identical fixed workload, times both with NVIDIA Nsight Systems,
and scores your throughput as a **fraction of cuBLAS** on the same GPU. So the
way to win is simply: produce the correct product, fast.

**Optimize for the NVIDIA A40** (Ampere, GA102; FP32 peak ~37 TFLOP/s, 48 GB
GDDR6, 84 SMs). Use what that architecture rewards — e.g. shared-memory tiling
with register blocking, coalesced global loads, avoiding shared-memory bank
conflicts, and enough occupancy to hide latency. Do your best to push throughput
on this specific GPU.

Constraints:
- Keep it a GPU implementation. Do **not** replace it with a CPU-only or naive
  fallback — both correctness *and* a real GPU performance floor are graded.
- You are free to edit **any** file (split the kernel across files, change the
  build, retune tiles — whatever helps). The one fixed contract is the entry
  point the grader calls:
  `void launchGemm(const float* dA, const float* dB, float* dC, int N)` in
  `src/gemm.cuh` — it must keep that exact signature and compute `C = A*B` for
  NxN row-major device matrices. Change it and the grader can't call your kernel
  (you forfeit the performance score; correctness still counts).
- Build, run, and iterate until correctness passes and your kernel is as fast as
  you can get it.

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
