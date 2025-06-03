include(FetchContent)

FetchContent_Declare(
        Catch2
        URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.tar.gz
        URL_HASH SHA256=122928b814b75717316c71af69bd2b43387643ba076a6ec16e7882bfb2dfacbb
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# 设置Catch2构建选项
set(CATCH_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(CATCH_INSTALL_EXTRAS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(Catch2)

# 设置目标属性
if (TARGET Catch2)
    set_target_properties(Catch2 Catch2WithMain PROPERTIES
            FOLDER "third_party"
    )
endif ()
