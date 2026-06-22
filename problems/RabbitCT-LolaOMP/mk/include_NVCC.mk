CC   = nvcc
LD = $(CC)

ifeq ($(strip $(ENABLE_OPENMP)),true)
OPENMP   = -Xcompiler "-fopenmp"
endif

VERSION  = --version
CFLAGS   = -O3 -std=c++17 $(CUDA_ARCH) $(OPENMP)
LFLAGS   = $(OPENMP)

DEFINES  +=  -DENABLE_CUDA -D_GNU_SOURCE -DRUNTIME_BACKEND_IS_CUDA
INCLUDES  =
LIBS      = -lcudart
