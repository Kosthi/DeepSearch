cmake_minimum_required(VERSION 3.12)

set(UNIT_TEST_SOURCES
        unit/core/test_interfaces.cpp
        unit/core/test_config.cpp
        unit/core/test_memory_manager.cpp
        unit/core/test_exceptions.cpp
        unit/distance/test_distance_computers.cpp
        unit/simd/test_simd_utils.cpp
        unit/simd/test_distance_functions.cpp
        unit/quantization/test_quantizers.cpp
        unit/searcher/test_searcher.cpp
        test_precision_integration.cpp
        test_main_flow_integration.cpp
)

set(BENCHMARK_TEST_SOURCES
        unit/simd/test_simd_benchmark.cpp
)

# 创建普通测试可执行文件
if (BUILD_TESTS)
    add_executable(deepsearch_tests
            ${UNIT_TEST_SOURCES}
    )

    # 链接库
    target_link_libraries(deepsearch_tests
            PRIVATE
            Catch2::Catch2WithMain
            deepsearch_core
            deepsearch_distance
            deepsearch_simd
            deepsearch_quantization
            deepsearch_graph
            deepsearch_searcher
    )

    # 设置包含目录
    target_include_directories(deepsearch_tests
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/tests
    )

    # 注册测试
    include(CTest)
    include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
    catch_discover_tests(deepsearch_tests)
endif ()

# 创建benchmark测试可执行文件
if (BUILD_BENCHMARKS)
    add_executable(deepsearch_benchmarks
            ${BENCHMARK_TEST_SOURCES}
    )

    # 链接库
    target_link_libraries(deepsearch_benchmarks
            PRIVATE
            Catch2::Catch2WithMain
            deepsearch_core
            deepsearch_distance
            deepsearch_simd
    )

    # 设置包含目录
    target_include_directories(deepsearch_benchmarks
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/tests
    )
endif ()
