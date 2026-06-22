# tiled-gemm

A shared-memory **tiled GEMM** (general matrix–matrix multiply, `C = A * B`) in
CUDA — the *dense linear algebra* HPC dwarf. This is a debugging task: the code
builds and runs, but the bundled correctness check fails for some matrix sizes.

## Build

```bash
make
```

Produces the `gemm` executable. Override the toolchain or target GPU if needed:

```bash
make NVCC=nvcc ARCH="-arch=sm_80"
```

## Run

```bash
./gemm        # demo / timed run only
./gemm -c     # run the correctness verification (use this)
```

`-c` multiplies several matrices on the GPU, compares each result against a
double-precision CPU reference, and prints per-size errors plus a final:

```
Mean Squared Error: <worst-case over all sizes>
```

A correct implementation yields a near-zero error for every size.

## Layout

| File            | Role                                                        |
| --------------- | ----------------------------------------------------------- |
| `src/gemm.cu`   | The tiled kernel and its launch — the code under test.      |
| `src/gemm.cuh`  | `launchGemm` interface.                                      |
| `src/main.cu`   | Fixed verification harness (sizes, CPU reference, MSE).     |
| `Makefile`      | `nvcc` build.                                               |
