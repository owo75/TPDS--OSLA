# ====================================================================
# Makefile for building the DBQR library and its test executable
# (最终正确版)
# ====================================================================

# --- 1. 编译器和通用标志 ---
NVCC := nvcc
NVCCFLAGS := -O3 -std=c++17 -Xcompiler -fPIC # -fPIC is good practice for libraries

# --- 2. GPU 架构配置 ---
GPU_ARCH_FLAGS := \
	-gencode arch=compute_80,code=sm_80   # For NVIDIA A100
	-gencode arch=compute_89,code=sm_89   # For NVIDIA RTX 4090
	-gencode arch=compute_90,code=sm_90   # For NVIDIA H100

# --- 3. 目录和公共库 ---
INCLUDES := -Iinc -Iutils/inc
LDLIBS := -lcublas -lcusolver -lcurand

# --- 4. [已配置] GEMMul8 库的特定配置 ---
GEMMUL8_INCLUDE := -I/home/sl/nfs_data/sl/GEMMul8/GEMMul8/include
GEMMUL8_LIBS := -L/home/sl/nfs_data/sl/GEMMul8/GEMMul8/lib -lgemmul8

# --- 5. 宏定义选项 ---
DEFINES :=
ifeq ($(USE_MGS), 1)
	DEFINES += -DUSE_MGS
endif

# --- 6. 最终编译标志 ---
FULL_NVCCFLAGS := $(NVCCFLAGS) $(GPU_ARCH_FLAGS) $(DEFINES) $(INCLUDES)

# --- 7. [核心部分] 定义库和可执行文件 ---

# 目标可执行文件名
TARGET_EXEC := run_dbqr_test

# 目标静态库文件名
TARGET_LIB := libdbqr.a

# 构建库(libdbqr.a)所需要的所有源文件
# (根据您的文件结构，这应该包含了所有算法逻辑)
DBQR_LIB_SRCS := \
	src/DBQR.cu \
	src/TSQR.cu \
	src/ReconstructWY.cu \
	utils/myutils.cu

# 自动将 .cu 源文件列表转换为 .o 目标文件列表
DBQR_LIB_OBJS := $(DBQR_LIB_SRCS:.cu=.o)

# 测试程序(test_main.cu)的源文件和目标文件
TEST_MAIN_SRCS := test_main.cu
TEST_MAIN_OBJS := $(TEST_MAIN_SRCS:.cu=.o)

# --- 8. 构建规则 ---

# 默认目标：输入 `make` 时，只构建我们的测试程序
all: $(TARGET_EXEC)

# 规则 1: 如何链接最终的可执行文件 (TARGET_EXEC)
# 它依赖于测试程序的 .o 文件和我们创建的静态库 .a 文件
$(TARGET_EXEC): $(TEST_MAIN_OBJS) $(TARGET_LIB)
	@echo "--> Linking final executable: $@"
	$(NVCC) $(FULL_NVCCFLAGS) $^ -o $@ $(GEMMUL8_INCLUDE) $(LDLIBS) $(GEMMUL8_LIBS)

# 规则 2: 如何创建静态库 (TARGET_LIB)
# 它依赖于库所需的所有 .o 文件
$(TARGET_LIB): $(DBQR_LIB_OBJS)
	@echo "--> Creating static library: $@"
	ar rcs $@ $^

# 规则 3: 如何将任何 .cu 文件编译成 .o 文件 (通用模式规则)
# 这会处理根目录、src/ 和 utils/ 目录下的所有 .cu 文件
%.o: %.cu
	@echo "--> Compiling $<"
	$(NVCC) $(FULL_NVCCFLAGS) -c $< -o $@

src/%.o: src/%.cu
	@echo "--> Compiling $<"
	$(NVCC) $(FULL_NVCCFLAGS) -c $< -o $@

utils/%.o: utils/%.cu
	@echo "--> Compiling $<"
	$(NVCC) $(FULL_NVCCFLAGS) -c $< -o $@

# --- 9. 清理规则 ---
.PHONY: all clean

clean:
	@echo "Cleaning up generated files..."
	rm -f $(TARGET_EXEC) $(TARGET_LIB) $(DBQR_LIB_OBJS) $(TEST_MAIN_OBJS)
	@echo "Cleanup complete."