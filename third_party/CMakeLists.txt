cmake_minimum_required(VERSION 3.15)

# 设置FetchContent策略
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

# 包含各个库的配置文件
include(fmt/fmt.cmake)
include(spdlog/spdlog.cmake)
include(nlohmann_json/nlohmann_json.cmake)

if (BUILD_TESTS OR BUILD_BENCHMARKS)
    include(catch2/catch2.cmake)
endif ()

# 创建统一的接口库
add_library(deepsearch_third_party INTERFACE)

# 链接所有启用的库
if (TARGET fmt::fmt)
    target_link_libraries(deepsearch_third_party INTERFACE fmt::fmt)
endif ()

if (TARGET spdlog::spdlog)
    target_link_libraries(deepsearch_third_party INTERFACE spdlog::spdlog)
endif ()

if (TARGET nlohmann_json::nlohmann_json)
    target_link_libraries(deepsearch_third_party INTERFACE nlohmann_json::nlohmann_json)
endif ()

# 设置IDE文件夹组织
if (TARGET fmt)
    set_target_properties(fmt PROPERTIES FOLDER "third_party")
endif ()
if (TARGET spdlog)
    set_target_properties(spdlog PROPERTIES FOLDER "third_party")
endif ()
