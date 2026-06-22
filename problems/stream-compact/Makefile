NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++14
# Optional: override to target a specific GPU, e.g. ARCH="-arch=sm_80"
ARCH      ?=

compact: src/main.cu src/compact.cu src/compact.cuh
	$(NVCC) $(NVCCFLAGS) $(ARCH) src/main.cu src/compact.cu -o compact

clean:
	rm -f compact

.PHONY: clean
