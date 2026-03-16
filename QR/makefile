NVCC ?= /usr/local/cuda-12.8/bin/nvcc
CC = gcc

CUDA_ARCH_LIST ?= 80 89 90
GENCODE_FLAGS := $(foreach arch,$(CUDA_ARCH_LIST),-gencode arch=compute_$(arch),code=sm_$(arch))
MAGMA_ROOT ?= /home/sl/nfs_data/sl/magma-2.9.0/magma_install
MAGMA_INC ?= $(MAGMA_ROOT)/include
MAGMA_LIB ?= $(MAGMA_ROOT)/lib
MAGMA_DEFS ?=
GEMMUL8_ROOT ?= /home/sl/nfs_data/OSLA/GEMMul8/GEMMul8
GEMMUL8_LIB ?= $(GEMMUL8_ROOT)/lib
CFLAGS = $(GENCODE_FLAGS) -g -I/usr/local/cuda-12.8/include -I$(MAGMA_INC)
LFLAGS = -g -L/usr/local/cuda-12.8/lib64 -lcublas -lcublasLt -lcurand -lcudart -lcuda -lcusolver -L$(GEMMUL8_LIB) -lgemmul8

# Add .cu files here. Only maintain this list.
CU_FILES = DBChol.cu DBCholsyrk.cu DBCholgemm.cu FP64doubleleftcholesky.cu dblk_leftchol_GEMMul8.cu

# Derived targets and object files
TARGETS = $(basename $(CU_FILES))
OBJS = $(TARGETS:=.o)

.PHONY: all clean list

all: $(TARGETS)

list:
	@echo "CU_FILES: $(CU_FILES)"
	@echo "TARGETS : $(TARGETS)"
	@echo "ARCHS   : $(CUDA_ARCH_LIST)"
	@echo "MAGMA_INC: $(MAGMA_INC)"
	@echo "MAGMA_DEFS: $(MAGMA_DEFS)"

# Build executable from corresponding .cu file
%: %.cu
	$(NVCC) $(CFLAGS) -c $< -o $@.o
	$(NVCC) $(LFLAGS) $@.o -o $@

# MAGMA benchmark target
magma: magma.cu
	$(NVCC) $(CFLAGS) $(MAGMA_DEFS) -c $< -o $@.o
	$(NVCC) $(LFLAGS) $@.o -L$(MAGMA_LIB) -Xlinker -rpath -Xlinker $(MAGMA_LIB) -lmagma -o $@

clean:
	rm -f $(OBJS) $(TARGETS)
