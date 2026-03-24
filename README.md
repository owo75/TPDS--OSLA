# OSLA

OSLA is a collection of GPU-based dense matrix factorization implementations. The current repository mainly contains double-blocking implementations for:

- Cholesky factorization
- LU factorization
- QR factorization

This README summarizes the main source entry points for both the baseline versions and the GEMMul8-based versions, together with the basic build flow and command-line arguments.

## Directory Overview

```text
OSLA/
├── Cholesky/
│   ├── DBCholgemm.cu
│   └── dblk_leftchol_GEMMul8.cu
├── LU/
│   ├── DBLU/
│   │   └── hygon-osla-gertf/
│   │       ├── float/getrf.cu
│   │       └── double/getrf.cu
│   └── GEMMUL8/
│       └── hygon-osla-gertf/
│           ├── float/getrf.cu
│           └── double/getrf.cu
└── QR/
    └── DBQR/
        └── ori/
            └── src/
                ├── geqrf_single.cu
                ├── geqrf_double.cu
                ├── geqrf_single_gemmul8.cu
                └── geqrf_double_gemmul8.cu
```

## 1. Cholesky

There are currently two main Cholesky entry points:

- Baseline version: `/home/sl/nfs_data/OSLA/Cholesky/DBCholgemm.cu`
- GEMMul8 version: `/home/sl/nfs_data/OSLA/Cholesky/dblk_leftchol_GEMMul8.cu`

### Implementation Notes

- This is a double-blocking Cholesky factorization implementation.
- It supports both `float` and `double` in `DBCholgemm.cu`.
- It uses `cuBLAS`, `cuSOLVER`, and `cuRAND`.
- The input matrix is generated on the GPU and converted into an SPD matrix by symmetrization and diagonal adjustment.

More specifically:

- `DBCholgemm.cu` is the main Cholesky program and can switch to a GEMMul8 GEMM path through a runtime flag.
- `dblk_leftchol_GEMMul8.cu` is a separate GEMMul8-specific implementation.

### Command-Line Arguments

`DBCholgemm.cu`:

```bash
./dbchol n k nb [float|double] [--debug] [--eye] [--gemmul8] \
  [--gemmul8-moduli N] [--gemmul8-safe|--gemmul8-fast] \
  [--warmup N] [--runs N]
```

Argument description:

- `n`: matrix size
- `k`: outer blocking size
- `nb`: inner panel blocking size
- `float|double`: precision mode, default is `float`
- `--debug`: print backward error `||A - LL^T||_F / ||A||_F`
- `--eye`: use the identity matrix as input
- `--gemmul8`: enable the GEMMul8 path
- `--warmup N`: number of warm-up runs
- `--runs N`: number of benchmark runs

`dblk_leftchol_GEMMul8.cu`:

```bash
./dblk_leftchol_gemmul8 n k nb [--eye]
```

### Reference Build Commands

This part of the repository currently uses single-file entry points and does not provide a dedicated top-level `CMakeLists.txt` for these Cholesky programs. A direct `nvcc` build would typically look like this:

```bash
cd /home/sl/nfs_data/OSLA/Cholesky
nvcc -O3 -std=c++17 DBCholgemm.cu -o dbchol \
  -I../GEMMul8/GEMMul8/include \
  -lcublas -lcusolver -lcurand

nvcc -O3 -std=c++17 dblk_leftchol_GEMMul8.cu -o dblk_leftchol_gemmul8 \
  -I../GEMMul8/GEMMul8/include \
  -lcublas -lcusolver -lcurand
```

If your local GEMMul8 setup requires additional include directories, library directories, or link flags, add them accordingly.

### Example Runs

```bash
./dbchol 8192 4096 512 double --warmup 1 --runs 5
./dbchol 8192 4096 512 float --gemmul8 --runs 10
./dblk_leftchol_gemmul8 8192 4096 512 --eye
```

## 2. LU

The LU implementation is split into a baseline version and a GEMMul8 version. Each of them provides both single-precision and double-precision entry files.

Baseline entry files:

- float: `/home/sl/nfs_data/OSLA/LU/DBLU/hygon-osla-gertf/float/getrf.cu`
- double: `/home/sl/nfs_data/OSLA/LU/DBLU/hygon-osla-gertf/double/getrf.cu`

GEMMul8 entry files:

- float: `/home/sl/nfs_data/OSLA/LU/GEMMUL8/hygon-osla-gertf/float/getrf.cu`
- double: `/home/sl/nfs_data/OSLA/LU/GEMMUL8/hygon-osla-gertf/double/getrf.cu`

### Implementation Notes

- The float and double versions fix the data type in the source file.
- The implementation uses `cuBLAS` and `cuSOLVER`.
- Pivoting is supported.
- Comparison against `cuSOLVER getrf` is supported.
- Result verification is supported.
- The GEMMul8 version adds a GEMMul8-based update path.

### Build

Both the baseline and GEMMul8 directories contain their own `CMakeLists.txt`, so they can be built independently.

Baseline float:

```bash
cd /home/sl/nfs_data/OSLA/LU/DBLU/hygon-osla-gertf/float
mkdir -p build
cmake -S . -B build
cmake --build build -j
```

Baseline double:

```bash
cd /home/sl/nfs_data/OSLA/LU/DBLU/hygon-osla-gertf/double
mkdir -p build
cmake -S . -B build
cmake --build build -j
```

GEMMul8 float:

```bash
cd /home/sl/nfs_data/OSLA/LU/GEMMUL8/hygon-osla-gertf/float
mkdir -p build
cmake -S . -B build
cmake --build build -j
```

GEMMul8 double:

```bash
cd /home/sl/nfs_data/OSLA/LU/GEMMUL8/hygon-osla-gertf/double
mkdir -p build
cmake -S . -B build
cmake --build build -j
```

All of these build directories generate an executable named `getrf`.

### Command-Line Arguments

Common CLI pattern:

```bash
./getrf <n>
./getrf <n> <k> <nb>
./getrf <n> <k> <nb> [-p] [-c] [-v] [-d]
```

Argument description:

- `n`: matrix size
- `k`: accumulated blocking size
- `nb`: panel block size
- `-p`: enable pivoting
- `-c`: compare against cuSOLVER
- `-v`: verify the result
- `-d`: enable debug output

Additional GEMMul8-version arguments:

- `-s`: use the cuBLAS swap path
- `--use-int8gemm`: enable the int8 GEMM path in the GEMMul8 double-precision version

When only `n` is provided, the program automatically sets:

- `k = n / 2`
- `nb = k / 2`
- pivoting enabled by default
- cuSOLVER comparison enabled by default

### Example Runs

Baseline float:

```bash
cd /home/sl/nfs_data/OSLA/LU/DBLU/hygon-osla-gertf/float/build
./getrf 8192 4096 2048 -p -c
```

Baseline double:

```bash
cd /home/sl/nfs_data/OSLA/LU/DBLU/hygon-osla-gertf/double/build
./getrf 8192 4096 2048 -p -c -v
```

GEMMul8 float:

```bash
cd /home/sl/nfs_data/OSLA/LU/GEMMUL8/hygon-osla-gertf/float/build
./getrf 8192 4096 2048 -p -c -s
```

GEMMul8 double:

```bash
cd /home/sl/nfs_data/OSLA/LU/GEMMUL8/hygon-osla-gertf/double/build
./getrf 8192 4096 2048 -p -c --use-int8gemm
```

## 3. QR

The QR implementation also provides both baseline and GEMMul8 versions.

Baseline entry files:

- double: `/home/sl/nfs_data/OSLA/QR/DBQR/ori/src/geqrf_double.cu`
- float: `/home/sl/nfs_data/OSLA/QR/DBQR/ori/src/geqrf_single.cu`

GEMMul8 entry files:

- double: `/home/sl/nfs_data/OSLA/QR/DBQR/ori/src/geqrf_double_gemmul8.cu`
- float: `/home/sl/nfs_data/OSLA/QR/DBQR/ori/src/geqrf_single_gemmul8.cu`

### Implementation Notes

- This is the original DBQR path in the repository.
- The implementation uses TSQR and Reconstruct-WY related steps.
- Single and double precision are provided as separate source files.
- The GEMMul8 versions are separate source files, not runtime switches in the baseline executables.
- The programs print stage-level timings such as:
  - `TSQR time`
  - `Reconstruct WY time`
  - `GEMM time`
  - `WY time`
  - `TailMatrix time`
  - `total time`
  - `tflops`

### Build

This directory already provides a `CMakeLists.txt`:

```bash
cd /home/sl/nfs_data/OSLA/QR/DBQR/ori
mkdir -p build
cmake -S . -B build
cmake --build build -j
```

Relevant generated executables:

- `geqrf_single`
- `geqrf_double`
- `geqrf_single_gemmul8`
- `geqrf_double_gemmul8`

### Command-Line Arguments

All of these executables use the same argument pattern:

```bash
./geqrf_single m n nb b
./geqrf_double m n nb b
./geqrf_single_gemmul8 m n nb b
./geqrf_double_gemmul8 m n nb b
```

Argument description:

- `m`: number of rows
- `n`: number of columns
- `nb`: iterative blocking size
- `b`: TSQR or recursive sub-block size

### Example Runs

Baseline float:

```bash
cd /home/sl/nfs_data/OSLA/QR/DBQR/ori/build
./geqrf_single 32768 8192 1024 256
```

Baseline double:

```bash
cd /home/sl/nfs_data/OSLA/QR/DBQR/ori/build
./geqrf_double 32768 8192 1024 256
```

GEMMul8 float:

```bash
cd /home/sl/nfs_data/OSLA/QR/DBQR/ori/build
./geqrf_single_gemmul8 32768 8192 1024 256
```

GEMMul8 double:

```bash
cd /home/sl/nfs_data/OSLA/QR/DBQR/ori/build
./geqrf_double_gemmul8 32768 8192 1024 256
```

## Dependencies

Based on the current source code and build scripts, the minimum required environment is:

- CUDA Toolkit
- `nvcc`
- `cuBLAS`
- `cuSOLVER`
- `cuRAND`
- CMake 3.25 or newer for the LU and QR subprojects

If you want to enable GEMMul8-based paths, you also need a local GEMMul8 installation with the required headers and libraries available.

## Notes

- This top-level README only summarizes the main implementations discussed above.
- The repository also contains additional variants, scripts, and comparison programs, including MAGMA-based versions, GEMMul8 variants, and check programs.
- If needed, this README can be extended further with performance tables, directory responsibilities, algorithm summaries, and more detailed build instructions.
