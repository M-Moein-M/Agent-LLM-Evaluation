CC   = icx
GCC  = gcc
LINKER = $(CC)

ifeq ($(ENABLE_OPENMP),true)
OPENMP   = -qopenmp
endif

VERSION  = --version
CFLAGS   =  -O3 -xHost -std=c99 $(OPENMP)
#CFLAGS   += -qopt-zmm-usage=high #on CPUs with AVX512 support
LFLAGS   = $(OPENMP)
DEFINES  = -D_GNU_SOURCE
INCLUDES =
LIBS     =
