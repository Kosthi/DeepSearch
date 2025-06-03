# 创建 hnsw 库
add_library(deepsearch_graph STATIC
        hnsw_builder.cpp
        builder_factory.cpp  # 新增这一行
        graph.cpp
)

# 公开头文件路径（允许父目录或其他模块访问）
target_include_directories(deepsearch_graph
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}  # 头文件路径指向 hnsw 目录
)

# 链接第三方库
target_link_libraries(deepsearch_graph PUBLIC
        deepsearch_third_party
)

if (APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")

    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")

    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif ()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    target_link_libraries(deepsearch_graph PUBLIC OpenMP::OpenMP_CXX)
else ()
    message(WARNING "OpenMP not found, continuing without it")
endif ()
