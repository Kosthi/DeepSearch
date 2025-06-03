cmake_minimum_required(VERSION 3.12)

# 创建core库
add_library(deepsearch_core STATIC
    config.cpp
    logger.cpp
)

# 设置头文件包含路径
target_include_directories(deepsearch_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# 链接第三方库
target_link_libraries(deepsearch_core PUBLIC
    deepsearch_third_party
)

# 设置C++标准
target_compile_features(deepsearch_core PUBLIC cxx_std_17)

# 添加编译选项
target_compile_options(deepsearch_core PRIVATE
    -Wall -Wextra -O2
)
