cmake_minimum_required(VERSION 3.13)

project(CLIP_DEMO VERSION 1.0)
set(CMAKE_CXX_STANDARD 11)

# 设置cuda语言和CUDA架构
enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(CUDA 11.4 REQUIRED)
find_package(OpenCV REQUIRED)

# 设置项目根目录变量
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_LIST_DIR})

# 设置CMake的输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 设置公共的包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 添加子目录
add_subdirectory(third_party) # 如果需要自行编译第三方库
add_subdirectory(external)    # 使用FetchContent获取并构建外部依赖

# add TensorRT8
include_directories(/home/TensorRT-8.6.1.6/include)
include_directories(/home/TensorRT-8.6.1.6/samples/common)

set(TENSORRT_LIB_PATH "/home/TensorRT-8.6.1.6/lib")
file(GLOB LIBS "${TENSORRT_LIB_PATH}/*.so")

# 添加源代码目录
add_subdirectory(src)



