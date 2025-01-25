# Compiler settings
NVCC       = nvcc
CXX        = g++  # Use g++ for C++ files
CXXFLAGS   = -Wall -Wextra -O3  # Changed from CFLAGS to CXXFLAGS for C++
CUDAFLAGS  = -arch=sm_60 -Xcompiler="-Wall -Wextra -O3" # Adjust sm version to your GPU
LDFLAGS    = -lm
TARGET     = bitonic_sort_cuda

# Directories
SRC_DIR = src
INC_DIR = inc
OBJ_DIR = obj

# Source files
CXX_SOURCES = $(SRC_DIR)/utils.cpp $(SRC_DIR)/evaluations.cpp $(SRC_DIR)/main.cpp
CUDA_SOURCES = $(SRC_DIR)/bitonic_sort_v0.cu $(SRC_DIR)/bitonic_sort_v1.cu $(SRC_DIR)/bitonic_sort_v2.cu

# Object files
CXX_OBJECTS = $(CXX_SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
OBJECTS = $(CXX_OBJECTS) $(CUDA_OBJECTS)

# Include directories
INCLUDES = -I$(INC_DIR)

# Build rules
all: $(OBJ_DIR) $(TARGET)

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(TARGET): $(OBJECTS)
	$(NVCC) $(OBJECTS) -o $@ $(LDFLAGS)

# Compile C++ source files with g++
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Compile CUDA source files with nvcc
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(CUDAFLAGS) $(INCLUDES) -c $< -o $@

# Clean up
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean
