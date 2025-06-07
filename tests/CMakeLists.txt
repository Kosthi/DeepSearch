cmake_minimum_required(VERSION 3.12)

# 收集所有测试源文件
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

# 创建主要测试可执行文件
if (BUILD_TESTS)
    add_executable(deepsearch_tests
            ${UNIT_TEST_SOURCES}
    )

    # 链接所有必要的库
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

    # 覆盖率链接配置
    if(ENABLE_COVERAGE)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            target_link_libraries(deepsearch_tests PRIVATE gcov)
        endif()
        # 为覆盖率添加编译选项
        target_compile_options(deepsearch_tests PRIVATE --coverage -fprofile-arcs -ftest-coverage -O0 -g)
    endif()

    # 注册测试
    include(CTest)
    include(${Catch2_SOURCE_DIR}/extras/Catch.cmake)
    catch_discover_tests(deepsearch_tests)
    
    # 添加自定义测试标签
    set_tests_properties(${deepsearch_tests} PROPERTIES
        LABELS "unit;integration"
    )
endif ()

# 基准测试（可选）
if (BUILD_BENCHMARKS)
    add_executable(deepsearch_benchmarks
            ${BENCHMARK_TEST_SOURCES}
    )

    target_link_libraries(deepsearch_benchmarks
            PRIVATE
            Catch2::Catch2WithMain
            deepsearch_core
            deepsearch_distance
            deepsearch_simd
    )

    target_include_directories(deepsearch_benchmarks
            PRIVATE
            ${CMAKE_SOURCE_DIR}/src
            ${CMAKE_SOURCE_DIR}/tests
    )
endif ()
