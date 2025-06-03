include(FetchContent)

FetchContent_Declare(
        spdlog
        URL https://github.com/gabime/spdlog/archive/refs/tags/v1.12.0.tar.gz
        URL_HASH SHA256=4dccf2d10f410c1e2feaff89966bfc49a1abb29ef6f08246335b110e001e09a9
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# 设置spdlog构建选项
set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SPDLOG_BUILD_BENCH OFF CACHE BOOL "" FORCE)
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "" FORCE)  # 使用外部fmt
set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(spdlog)

# 设置目标属性
if (TARGET spdlog)
    set_target_properties(spdlog PROPERTIES
            FOLDER "third_party"
            POSITION_INDEPENDENT_CODE ON
    )
endif ()
