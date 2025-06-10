# 创建 deepsearch_index 静态库
add_library(deepsearch_index STATIC)

# 设置源文件和头文件
target_sources(deepsearch_index PRIVATE
        ivf_index.cpp
        hnsw_index.cpp
        index_factory.cpp
        graph_index.cpp
)

# 设置头文件
set(INDEX_HEADERS
        ivf_index.h
        hnsw_index.h
        index_factory.h
        index.h
        graph_index.h
)

# 设置包含目录
target_include_directories(deepsearch_index
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接依赖库
target_link_libraries(deepsearch_index PRIVATE
        deepsearch_core
        deepsearch_quantization
        deepsearch_simd
        deepsearch_distance
        deepsearch_searcher
)

# 设置编译选项
target_compile_options(deepsearch_index PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -O3 -march=native>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# 设置 C++ 标准
set_target_properties(deepsearch_index PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

# 平台特定设置
if (APPLE)
    target_compile_definitions(deepsearch_index PRIVATE __APPLE__)
elseif (UNIX)
    target_compile_definitions(deepsearch_index PRIVATE __LINUX__)
elseif (WIN32)
    target_compile_definitions(deepsearch_index PRIVATE __WINDOWS__)
endif ()

# SIMD 支持
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(deepsearch_index PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-msse4.2 -mavx -mavx2>
            $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
    )
endif ()

# 安装头文件
install(FILES ${INDEX_HEADERS}
        DESTINATION include/deepsearch/index
)
