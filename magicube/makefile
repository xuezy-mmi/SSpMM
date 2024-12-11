NVCC = /usr/local/cuda/bin/nvcc
# SPUTNIK_PATH = /home/xuezeyu/sputnik
NVCC_FLAGS = -std=c++11 -arch=sm_80 -lineinfo -lcublas -lcusparse  --ptxas-options=-v

##################################################################

## Project file structure ##

# Source file directory:
SRC_DIR = src

# Object file directory:
OBJ_DIR = bin

# Include header file directory
INC_DIR = include

# Library directory
LIB_DIR = /usr/local/cuda/lib64
# SPUTNIK_DIR = ${SPUTNIK_PATH}/build/sputnik

#output directory
OUT_DIR = out
#csv output directory
RECORD_DIR = data


##################################################################

## Compile ##

# sddmm_benchmark: $(OBJ_DIR)/sddmm_benchmark.o $(OBJ_DIR)/cuda_sddmm.o $(OBJ_DIR)/wmma_sddmm.o $(OBJ_DIR)/cublas_gemm.o
#	@$(NVCC) $(NVCC_FLAGS) $^ -o $@

spmm: $(OBJ_DIR)/spmm_test.o $(OBJ_DIR)/wmma_spmm.o
	@$(NVCC) $(NVCC_FLAGS) $^  -o $(OUT_DIR)/$@

rowmerge: $(OBJ_DIR)/spmm_rowmerge.o $(OBJ_DIR)/wmma_spmm.o
	@$(NVCC) $(NVCC_FLAGS) $^  -o $(OUT_DIR)/$@

# Compile main file to object file
$(OBJ_DIR)/%.o : %.cpp
	@$(NVCC) $(NVCC_FLAGS) -x c++ -c $< -o $@ 


# Compile CUDA source files to object files
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	@$(NVCC) $(NVCC_FLAGS) -x cu -c $< -o $@

spmm_run:
	$(OUT_DIR)/spmm paths.txt 512 8 2 0 0 0

rowmerge_run:
	$(OUT_DIR)/rowmerge paths.txt 512 8 1 0 0 0

clean:
	@rm -f $(OBJ_DIR)/*.o
