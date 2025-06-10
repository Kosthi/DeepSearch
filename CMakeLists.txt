cmake_minimum_required(VERSION 3.15)
project(DeepSearch)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译选项开关
option(BUILD_TESTS "Build tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)

# 代码覆盖率配置
if (ENABLE_COVERAGE)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -fprofile-arcs -ftest-coverage -O0 -g")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
        message(STATUS "Coverage reporting enabled")
    else ()
        message(WARNING "Coverage reporting only supported with GCC or Clang")
    endif ()
endif ()

# 全局添加 src 目录到头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)

# 添加子目录
add_subdirectory(src/core)
add_subdirectory(src/graph)
add_subdirectory(src/distance)
add_subdirectory(src/simd)
add_subdirectory(src/quantization)
add_subdirectory(src/searcher)
add_subdirectory(src/index)
add_subdirectory(src/builder)
add_subdirectory(third_party)

# 只有在开启测试时才添加测试目录
if (BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif ()

# 设置通用编译选项（所有平台共用）
if (ENABLE_COVERAGE)
    set(COMMON_FLAGS "-Wall -Wextra -O0 -g -fpic -fexceptions -fno-rtti")
else ()
    set(COMMON_FLAGS "-Wall -Wextra -O3 -fpic -fexceptions -fno-rtti")
endif ()

# 可执行文件配置
add_executable(DeepSearch
        src/main.cpp
        src/searcher/searcher.cpp
)

# 平台特定配置
if (APPLE)
    # macOS 配置
    set(CMAKE_CXX_FLAGS "${COMMON_FLAGS}")
    target_link_libraries(DeepSearch PRIVATE c++ c++abi)
else ()
    # Linux 配置
    set(CMAKE_CXX_FLAGS "${COMMON_FLAGS} -lrt -fopenmp -ftree-vectorize")

    # 仅 x86 架构添加 -march=native
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64|i[3-6]86)")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif ()
endif ()

# 链接库
target_link_libraries(DeepSearch
        PRIVATE
        deepsearch_index
        deepsearch_builder
        deepsearch_searcher
        fmt::fmt
        spdlog::spdlog
        nlohmann_json::nlohmann_json
)

# 覆盖率链接
if (ENABLE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_link_libraries(DeepSearch PRIVATE gcov)
endif ()
