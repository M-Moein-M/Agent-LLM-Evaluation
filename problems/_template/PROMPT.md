Implement `sum_squares` in C.

`src/solve.c` contains a working but slow function:

```c
double sum_squares(const double* x, long n);
```

It must return the sum of `x[i] * x[i]` for `i` in `[0, n)`. Make it as fast as you can
on this machine while keeping the result correct.

Constraints:
- Do **not** modify `src/main.c`. The input, the reference, and the error metric there
  are fixed and are how you are graded.
- Build with `make` and run `./prog -c` yourself. Iterate until the reported
  **Mean Squared Error** is ~0.

When finished, write a file `solution.json` in the problem root with the exact commands
you used:

```json
{
  "build": ["make"],
  "run": ["./prog -c"],
  "variant_name": "template"
}
```

Requirements:
- All commands must be relative to the problem root directory.
- The evaluation harness re-runs these commands independently for verification.
- If your solution does not compile or reports a non-zero Mean Squared Error, it fails.
