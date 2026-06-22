CC   = clang
LD = $(CC)

ifeq ($(ENABLE_OPENMP),true)
ifeq ($(shell uname -s),Darwin)
OPENMP   = -Xpreprocessor -fopenmp
LIBS     = -L/opt/homebrew/opt/libomp/lib -lomp
else
OPENMP   = -fopenmp
endif
endif

VERSION  = --version
CFLAGS   = -O3 -ffast-math -std=c99 $(OPENMP)
#CFLAGS   = -Ofast -fnt-store=aggressive  -std=c99 $(OPENMP) #AMD CLANG
LFLAGS   = $(OPENMP)
DEFINES  = -D_GNU_SOURCE
ifeq ($(shell uname -s),Darwin)
INCLUDES = -I/opt/homebrew/opt/libomp/include
else
INCLUDES =
endif
