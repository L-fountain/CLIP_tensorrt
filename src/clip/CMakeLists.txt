# 生成一个CLIP的动态库
add_library(CLIP SHARED)

set(CLIP_SOURCES
    clip_base.cpp
    clip_vision.cpp
    clip_text.cpp
)

# 添加源文件到库目标
target_sources(CLIP PUBLIC ${CLIP_SOURCES})

target_include_directories(CLIP PUBLIC 
    $<BUILD_INTERFACE:${PROJECT_ROOT_DIR}/include/my_demo>
    $<BUILD_INTERFACE:${PROJECT_ROOT_DIR}/include/my_demo/preprocess>
    $<BUILD_INTERFACE:${PROJECT_ROOT_DIR}/include/my_demo/clip>
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(CLIP PUBLIC ${CUDA_LIBRARIES} ${OpenCV_LIBS})