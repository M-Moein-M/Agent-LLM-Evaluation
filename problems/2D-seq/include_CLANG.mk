CC   = clang
GCC  = cc
LINKER = $(CC)

ifeq ($(ENABLE_OPENMP),true)
OPENMP   = -fopenmp
#OPENMP   = -Xpreprocessor -fopenmp #required on Macos with homebrew libomp
LIBS     = # -lomp
endif

VERSION  = --version
CFLAGS   = -O3 -ffast-math -std=c99 $(OPENMP)
LFLAGS   = $(OPENMP) -lm
DEFINES  = -D_GNU_SOURCE
INCLUDES =
