# 设置源文件
set(SEARCHER_SOURCES
        searcher.cpp
)

# 设置头文件
set(SEARCHER_HEADERS
        searcher.h
        searcher_impl.h
)

# 创建searcher库
add_library(deepsearch_searcher STATIC ${SEARCHER_SOURCES})

# 设置头文件包含路径
target_include_directories(deepsearch_searcher PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
)

# 链接依赖库
target_link_libraries(deepsearch_searcher
        PUBLIC
        deepsearch_core
        deepsearch_quantization
        deepsearch_simd
        deepsearch_graph
)

# 设置C++标准
set_target_properties(deepsearch_searcher PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# 编译选项
target_compile_options(deepsearch_searcher PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wall -Wextra -O3>
        $<$<CXX_COMPILER_ID:Clang>:-Wall -Wextra -O3>
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# 平台特定配置
if (APPLE)
    # macOS特定配置
    target_compile_definitions(deepsearch_searcher PRIVATE APPLE_PLATFORM=1)
else ()
    # Linux特定配置
    target_compile_options(deepsearch_searcher PRIVATE -fopenmp)
    target_link_libraries(deepsearch_searcher PRIVATE rt)
endif ()

# 安装配置
install(TARGETS deepsearch_searcher
        EXPORT DeepSearchTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)

install(FILES ${SEARCHER_HEADERS}
        DESTINATION include/deepsearch/searcher
)
