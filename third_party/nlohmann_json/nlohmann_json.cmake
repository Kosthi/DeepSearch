include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/archive/refs/tags/v3.11.3.tar.gz
    URL_HASH SHA256=4842187627440306124040909540027732298717808266205807185381831037
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# 设置nlohmann_json构建选项
set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
set(JSON_Install OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(nlohmann_json)
