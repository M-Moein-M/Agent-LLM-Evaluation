CC  = icc
CXX  = icpc
AS  = icc

PAS = ./perl/AsmGen.pl 
ISPC = $(HOME)/ispc-v1.6.0-linux/ispc

#DEBUG=-g

ANSI_CFLAGS += -std=c99
#ANSI_CFLAGS += -pedantic
#ANSI_CFLAGS += -Wextra

CFLAGS   =  -Wno-format -vec-report=0 -fpic -openmp -std=c99 $(DEBUG) $(MIC) -O3 -fno-alias
CXXFLAGS =  -Wno-format -vec-report=0 -fpic -openmp $(DEBUG) $(MIC) -O3 -fno-alias
ASFLAGS  = -c $(DEBUG) $(MIC)
CPPFLAGS = $(DEBUG) $(MIC)
ISPCFLAGS = $(DEBUG)
LFLAGS   = -shared -openmp $(DEBUG) $(MIC)
DEFINES  = -D_GNU_SOURCE
DEFINES   += -DMAX_NUM_THREADS=128
ifdef MIC
DEFINES   += -DVECTORSIZE=16
else
DEFINES   += -DVECTORSIZE=8
endif
#DEFINES  += -DCODE_ANALYSER
#DEFINES  += -DLIKWID_PERFMON
#DEFINES  += -DKERNEL_CYCLES


INCLUDES = -I../includes
#LIBS     = -L/home/<group>/<user>/new-likwid/mic/lib/ -llikwid
