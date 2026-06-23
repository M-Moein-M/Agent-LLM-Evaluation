Implement **order-preserving stream compaction** on the GPU.

`src/compact.cu` contains an unimplemented function:

```c
int compact(const int* d_in, int n, int threshold, int* d_out);
```

It must copy every element of `d_in` (length `n`, device memory) whose value is
**strictly greater than** `threshold` into `d_out`, **preserving the elements'
original relative order**, and return the number of elements written. The kept
elements must occupy `d_out[0 .. count)` in the same order they appear in
`d_in`.

This runs on an **NVIDIA A40** (Ampere, GA102; 48 GB GDDR6, 84 SMs). Beyond
correctness, prefer an efficient implementation for that GPU — coalesced memory
access, a work-efficient parallel scan, and enough occupancy to hide latency.

Constraints:
- Implement it yourself with your own CUDA kernels. Do **not** use Thrust or CUB
  (e.g. `thrust::copy_if`, `cub::DeviceSelect`) — the point is the parallel
  algorithm.
- Do **not** modify the verification harness in `src/main.cu`. The input, the
  reference, the number of repetitions, and the error metric there are fixed and
  are how you are graded.
- Build and run `./compact -c` yourself and iterate until the reported **Mean
  Squared Error** is ~0 (a correct solution scores exactly 0).

When finished, write a file `solution.json` in the problem root with the exact
commands you used:

```json
{
  "build": ["make"],
  "run": ["./compact -c"],
  "variant_name": "stream-compact"
}
```

Requirements:
- All commands must be relative to the problem root directory.
- These commands will be independently re-run by the evaluation harness for
  verification.
- If your solution does not compile or reports a non-zero Mean Squared Error,
  the harness will fail — write commands that actually work.
