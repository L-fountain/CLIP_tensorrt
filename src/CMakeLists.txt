# 创建PREPROCESS子组件库
add_subdirectory(preprocess)

# 创建CLIP子组件库
add_subdirectory(clip)
 
set(SAMPLES_COMMON_SOURCES "/home/TensorRT-8.6.1.6/samples/common/logger.cpp")


# 将子组件链接到主程序
cuda_add_executable(CLIP_DEMO main.cpp ${SAMPLES_COMMON_SOURCES})

# link
target_link_libraries(CLIP_DEMO ${LIBS} ${CUDA_LIBRARIES} ${OpenCV_LIBS} CLIP PREPROCESS)