# Supported: GCC, CLANG, NVCC
TOOLCHAIN ?= GCC
ENABLE_OPENMP ?= false
ENABLE_LIKWID ?= false

# GPU architecture (only used when TOOLCHAIN=NVCC)
# -gencode=arch=compute_80,code=sm_80 # for A100
# -gencode=arch=compute_86,code=sm_86 # for A40
# -gencode=arch=compute_90,code=sm_90 # for GH200
CUDA_ARCH ?= -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86

# Feature options
OPTIONS +=  -DARRAY_ALIGNMENT=64
OPTIONS +=  -DMAX_NUM_THREADS=32
OPTIONS +=  -DVECTORSIZE=4
#OPTIONS +=  -DVERBOSE_AFFINITY
#OPTIONS +=  -DVERBOSE_DATASIZE
#OPTIONS +=  -DVERBOSE_TIMER
