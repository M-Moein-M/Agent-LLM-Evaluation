NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++14
# Optional: override to target a specific GPU, e.g. ARCH="-arch=sm_80"
ARCH      ?=

gemm: src/main.cu src/gemm.cu src/gemm.cuh
	$(NVCC) $(NVCCFLAGS) $(ARCH) src/main.cu src/gemm.cu -o gemm

clean:
	rm -f gemm

.PHONY: clean
