# 生成一个PREPROCESS的动态库
add_library(PREPROCESS SHARED)

set(PREPROCESS_SOURCES
    preprocess.cpp
)

set(CUDA_SOURCE_FILE preprocess.cu)
set_source_files_properties(${CUDA_SOURCE_FILE} PROPERTIES LANGUAGE CUDA)

# 添加源文件到库目标
target_sources(PREPROCESS PRIVATE ${PREPROCESS_SOURCES} ${CUDA_SOURCE_FILE})

target_include_directories(PREPROCESS PUBLIC
    $<BUILD_INTERFACE:${PROJECT_ROOT_DIR}/include/my_demo>
    $<BUILD_INTERFACE:${PROJECT_ROOT_DIR}/include/my_demo/preprocess>
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(PREPROCESS PUBLIC ${CUDA_LIBRARIES} ${OpenCV_LIBS})