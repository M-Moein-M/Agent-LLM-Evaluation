# RabbitCT

Fast 3D cone-beam reconstruction is mandatory for many clinical workflows.
Backprojection is a major component of many reconstruction algorithms: it
requires projecting each voxel onto the projection data, interpolating the
data, and updating the voxel value. This step is the bottleneck of most
reconstruction algorithms and has been the focus of optimization in many publications.

A crucial limitation of those publications is that the presented results are not
comparable, mainly due to variations in data acquisition, preprocessing, chosen
geometries, and the lack of a common publicly available test dataset. RabbitCT
provides an open platform for worldwide comparison of backprojection performance
on different architectures using one specific, clinical, high-resolution C-arm
CT dataset of a real rabbit. It includes a benchmark interface, prototype
implementations in C, and image quality measures.

RabbitCT was a collaboration of the Department of Neuroradiology and the Pattern
Recognition Lab at the Friedrich-Alexander-Universität Erlangen-Nürnberg.

Results are clustered by problem size (256, 512, or 1024; 128 is not considered
in the ranking), denoting the edge length of the reconstructed cubic volume in
voxels.

This repository provides a single reference implementation:

- **LolaBunny** -- simple reference implementation, straightforward triple-nested loop

## Build

### Prerequisites

- C compiler: GCC, Clang, or Intel ICX (x86-64 or AARCH64)
- Optional: [LIKWID](https://github.com/RRZE-HPC/likwid) (for hardware performance counter measurements)

### Configuration

Edit `config.mk` to select your toolchain and features:

```makefile
TOOLCHAIN    ?= CLANG          # GCC, CLANG, ICX
ENABLE_LIKWID ?= false         # set to true for LIKWID marker API
```

Compiler-specific flags are in the `mk/include_<TOOLCHAIN>.mk` files and can be
adjusted if needed.

### Compile

```txt
make
```

This produces the executable `rabbitRunner-<TOOLCHAIN>` (e.g. `rabbitRunner-CLANG`).

Other targets:

```txt
make clean       # remove object files
make distclean   # remove all build artifacts and the executable
make asm         # generate assembly listings
make format      # Reformat all source files (requires a recent clang-format in path)
```

## Usage

```txt
./rabbitRunner-CLANG -i <input> -m <algorithm> -s <size> [OPTIONS]
```

### Required flags

| Flag | Argument | Description                                                     |
| ---- | -------- | --------------------------------------------------------------- |
| `-i` | filename | Input CT projection data file (`.rct`)                          |
| `-m` | name     | Algorithm to use (run with `-h` to list available)              |
| `-s` | size     | Problem size (volume dimension): `128`, `256`, `512`, or `1024` |

### Optional flags

| Flag | Argument | Description                                                        |
| ---- | -------- | ------------------------------------------------------------------ |
| `-b` | count    | Number of projections to buffer per iteration (default: 1)         |
| `-c` | filename | Reference volume for result verification                           |
| `-o` | filename | Write reconstructed volume as raw binary float                     |
| `-p` | filename | Write middle axial slice as 16-bit PGM image                       |
| `-v` |          | Verbose output                                                     |
| `-h` |          | Print help message                                                 |

### Problem sizes and voxel resolution

| Size | Volume dimensions  | Voxel size |
| ---- | ------------------ | ---------- |
| 128  | 128 x 128 x 128    | 2.0 mm     |
| 256  | 256 x 256 x 256    | 1.0 mm     |
| 512  | 512 x 512 x 512    | 0.5 mm     |
| 1024 | 1024 x 1024 x 1024 | 0.25 mm    |

### Examples

Simple reconstruction with the reference algorithm:

```bash
./rabbitRunner-CLANG \
  -i ./RabbitInput/RabbitInput.rct \
  -m LolaBunny \
  -s 256
```

Verification with a reference volume and slice output:

```bash
./rabbitRunner-CLANG \
  -i ./RabbitInput/RabbitInput.rct \
  -m LolaBunny \
  -s 256 \
  -c ./RabbitInput/Reference256.vol \
  -p slice.pgm
```

## Output and interpreting results

### Console output

Every run prints runtime statistics:

```txt
Runtime statistics:
Total: 12.345 s
Average: 34.29 ms
```

- **Total** -- wall-clock time for all backprojection iterations
- **Average** -- mean time per projection image (total / number of projections)

### Quality metrics (with `-c`)

When a reference volume is provided via `-c`, the reconstructed volume is compared against it voxel-by-voxel in Hounsfield Units (HU). The following metrics are printed:

```txt
Quality of reconstructed volume:
Root Mean Squared Error: 0.123 HU
Mean Squared Error: 0.0151 HU^2
Max. Absolute Error: 2.5 HU
PSNR: 84.3 db
mean time [usec]: 34291
```

| Metric                  | Meaning                                                                                                                                                                                                                          |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **RMSE**                | Root mean squared error across all voxels in HU. Lower is better. A value near 0 indicates the reconstruction matches the reference.                                                                                             |
| **MSE**                 | Mean squared error (RMSE squared). Useful for numerical comparisons.                                                                                                                                                             |
| **Max. Absolute Error** | Largest single-voxel deviation in HU. Shows worst-case accuracy.                                                                                                                                                                 |
| **PSNR**                | Peak signal-to-noise ratio in dB, computed as `10 * log10(4095^2 / MSE)`. Higher is better. Values above 80 dB indicate excellent reconstruction quality. A value of INF means the reconstruction is identical to the reference. |

For a correct implementation, RMSE should be very close to 0 and PSNR should be
very high. Non-zero errors typically come from floating-point rounding
differences between the implementation and the reference.

### Output files

#### PGM slice image (`-p`)

Writes the middle axial slice of the reconstructed volume as a 16-bit portable
graymap (PGM, format P5). Voxel values are converted from linear attenuation
coefficients to Hounsfield Units and clamped to the range [0, 4095].

This file can be viewed with standard image tools such as ImageJ, GIMP, or
`feh`. It provides a quick visual check that the reconstruction is producing
sensible results.

#### Raw volume (`-o`)

Writes the entire reconstructed volume as a flat binary file of 32-bit IEEE 754
floats. The file contains `size x size x size` float values in row-major order
(x varies fastest, then y, then z).

To view the volume, import it as raw data in a tool such as ParaView or ImageJ
(File > Import > Raw), specifying:

- Data type: 32-bit float
- Dimensions: size x size x size (e.g. 256 x 256 x 256)
- Little-endian byte order

#### Verification file (`result.rctd`)

Written automatically when `-c` is used. This binary file contains:

- Volume size and projection count
- Per-projection runtime (mean time, in microseconds)
- The central slice data (float values)
- Error metrics: MSE, RMSE, max error
- An error histogram with 11 bins (0-1, 1-2, ..., 9-10, >10 HU), showing the distribution of per-voxel absolute errors

This file is intended for automated comparison and result archival.

## Algorithm

### LolaBunny (reference)

A straightforward implementation of cone-beam backprojection. For each
projection, it iterates over every voxel in the volume, projects the voxel onto
the detector plane using the 3x4 projection matrix, performs bilinear
interpolation to sample the projection image, and accumulates the weighted value
into the volume.

## Input data

The input dataset is not included in this repository. To download it, run:

```bash
./download-input.sh
```

This downloads the `RabbitInput.tgz` archive (~2.9 GB), extracts it into the
`RabbitInput/` directory, and removes the archive.

The `RabbitInput/` directory contains the test dataset:

| File                 | Description                              |
| -------------------- | ---------------------------------------- |
| `RabbitInput.rct`    | Projection images and matrices           |
| `RabbitGeometry.rct` | Global geometry data (unused by LolaBunny) |
| `Reference128.vol`   | Reference volume for size 128            |
| `Reference256.vol`   | Reference volume for size 256            |
| `Reference512.vol`   | Reference volume for size 512            |
| `Reference1024.vol`  | Reference volume for size 1024           |

## License

MIT -- see [LICENSE](LICENSE).
