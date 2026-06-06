# stream-compact

Order-preserving **stream compaction** on the GPU — keep every array element
that passes a predicate (`value > threshold`), discard the rest, and keep the
survivors in their original order. A *map/filter with a data-dependent output
position*: where each kept element lands depends on how many kept elements
precede it, so this breaks the embarrassingly-parallel pattern and needs a
parallel scan rather than an independent per-element write.

This is a generation task: implement `compact()` in [src/compact.cu](src/compact.cu).

## Build

```bash
make                       # or: make ARCH="-arch=sm_80"
```

## Run

```bash
./compact        # timed run only
./compact -c     # run the correctness verification (use this)
```

`-c` runs your `compact()` against a double-checked CPU reference on a fixed
random permutation, **repeated many times**, and prints the worst-case:

```
Mean Squared Error: <worst over all repetitions>
```

A correct, order-preserving implementation scores exactly 0 on every
repetition. An implementation that produces the right elements in the wrong
(or nondeterministic) order shows a large error.

## Layout

| File              | Role                                                       |
| ----------------- | ---------------------------------------------------------- |
| `src/compact.cu`  | The function to implement — the code under test.           |
| `src/compact.cuh` | `compact()` interface and contract.                        |
| `src/main.cu`     | Fixed verification harness (input, CPU reference, metric). |
| `Makefile`        | `nvcc` build.                                              |
