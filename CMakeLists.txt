cmake_minimum_required(VERSION 3.15)
project(DeepSearch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译选项开关
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)

# 全局添加 src 目录到头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# 添加子目录
add_subdirectory(src/core)
add_subdirectory(src/graph)
add_subdirectory(src/distance)
add_subdirectory(src/simd)
add_subdirectory(src/quantization)
add_subdirectory(src/searcher)
add_subdirectory(third_party)

# 只有在开启测试时才添加测试目录
if (BUILD_TESTS)
    add_subdirectory(tests)
endif ()

# 设置通用编译选项（所有平台共用）
set(COMMON_FLAGS "-Wall -Wextra -O3 -fpic -fexceptions -fno-rtti")

# 可执行文件配置
add_executable(DeepSearch
        src/main.cpp
        src/searcher/searcher.cpp
)

# 平台特定配置
if (APPLE)
    # macOS 配置（移除 -fopenmp 选项，显式链接C++标准库）
    set(CMAKE_CXX_FLAGS "${COMMON_FLAGS}")
    target_link_libraries(DeepSearch PRIVATE c++ c++abi)
else ()
    # Linux 配置（保留 OpenMP 支持）
    set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} -lrt -fopenmp -ftree-vectorize")

    # 仅 x86 架构添加 -march=native
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|i[3-6]86)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif ()
endif ()

# 链接平台特定库
if (UNIX AND NOT APPLE)
    target_link_libraries(DeepSearch PRIVATE rt)
endif ()

# 链接内部库和三方库
target_link_libraries(DeepSearch
        PRIVATE
        deepsearch_graph
        deepsearch_searcher
        deepsearch_third_party  # 新增的三方库接口
)

# 只有在开启测试时才启用测试
if (BUILD_TESTS)
    enable_testing()
endif ()
