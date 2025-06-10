if (APPLE)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_C_LIB_NAMES "omp")

    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")

    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif ()

find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    # target_link_libraries(deepsearch_graph PUBLIC OpenMP::OpenMP_CXX)
else ()
    message(WARNING "OpenMP not found, continuing without it")
endif ()
