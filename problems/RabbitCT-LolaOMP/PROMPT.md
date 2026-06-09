Implement a GPU (CUDA) version of the RabbitCT LolaOMP backprojection.

Add a new GPU algorithm named `LolaOMPGPU` that produces the same output as LolaBunny and works with the existing runner and `-c` verification. Keep LolaBunny intact for baseline comparison.

The CPU OpenMP reference is in `src/LolaOMP.c`. Unlike LolaBunny, it uses a line-range clipping optimization — a preprocessing step (`computeLineRanges` in `src/analyseGeometry.c`) that, for each (projection, z, y), restricts the x-voxels to those whose rays actually hit the detector. This needs the global geometry file (`-a RabbitGeometry.rct`); `run-bench.sh` passes it automatically. Study both files.

Build with the CUDA toolchain, run, and verify your implementation. Debug until it produces correct results. Use size 256 and `./RabbitInput/Reference256.vol` for verification with the `-c` flag.

Use **CUDA only**. Do **not** use OpenCL — its runtime is not available in this environment and OpenCL device-query calls (e.g. `clGetPlatformIDs`) will hang.

Build:

```
make TOOLCHAIN=NVCC CUDA_ARCH="-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_80,code=sm_80"
```

Run and verify:

```
./run-bench.sh -s 256 -v LolaOMPGPU
```

When finished, write a file `solution.json` in the problem root with the exact commands you used:

```json
{
  "build": ["make TOOLCHAIN=NVCC CUDA_ARCH=\"-gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_80,code=sm_80\""],
  "run": ["./run-bench.sh -s 256 -v LolaOMPGPU"],
  "variant_name": "LolaOMPGPU"
}
```

Requirements:
- All commands must be relative to the problem root directory.
- `variant_name` is the name passed to `-v` in the run command.
- These commands will be independently re-run by the evaluation harness for verification.
- If your solution does not compile or produces incorrect results, the harness will fail — write commands that actually work.
