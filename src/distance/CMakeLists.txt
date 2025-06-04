# Distance computation module

set(DISTANCE_SOURCES
        computers.cpp
)

set(DISTANCE_HEADERS
        computers.h
)

# 创建距离计算库
add_library(deepsearch_distance ${DISTANCE_SOURCES})

# 设置包含目录
target_include_directories(deepsearch_distance PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(deepsearch_distance PUBLIC spdlog::spdlog)

# # 使用统一的 SIMD 配置函数
# if(SIMD_CONFIGURE_FUNCTION_AVAILABLE)
#     configure_simd_support(deepsearch_distance)
# endif()

# 设置C++标准
set_target_properties(deepsearch_distance PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
)

# 安装头文件
install(FILES ${DISTANCE_HEADERS}
        DESTINATION include/deepsearch/distance
)

# 安装库
install(TARGETS deepsearch_distance
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
