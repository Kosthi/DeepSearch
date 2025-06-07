include(FetchContent)

set(URL "https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz")
set(HASH "SHA256=0d8ef5af7f9794e3263480193c491549b2ba6cc74bb018906202ada498a79406")

# CMake ≥ 3.24 使用新特性，旧版本跳过
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    FetchContent_Declare(nlohmann_json
            URL ${URL}
            URL_HASH ${HASH}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
else ()
    FetchContent_Declare(nlohmann_json
            URL ${URL}
            URL_HASH ${HASH}
    )
endif ()

unset(URL)
unset(HASH)

# 设置nlohmann_json构建选项
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nlohmann_json)
