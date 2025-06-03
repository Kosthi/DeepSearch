include(FetchContent)

FetchContent_Declare(
        fmt
        URL https://github.com/fmtlib/fmt/archive/refs/tags/10.1.1.tar.gz
        URL_HASH SHA256=78b8c0a72b1c35e4443a7e308df52498252d1cefc2b08c9a97bc9ee6cfe61f8b
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# 设置fmt构建选项
set(FMT_DOC OFF CACHE BOOL "" FORCE)
set(FMT_TEST OFF CACHE BOOL "" FORCE)
set(FMT_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(fmt)

# 设置目标属性
if (TARGET fmt)
    set_target_properties(fmt PROPERTIES
            FOLDER "third_party"
            POSITION_INDEPENDENT_CODE ON
    )
endif ()
