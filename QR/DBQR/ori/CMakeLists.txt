cmake_minimum_required(VERSION 3.25)

project(geqrf_db_ori VERSION 0.1.0 LANGUAGES C CXX CUDA)

# Ensure CUDA imported targets (CUDA::cublas, etc.) are available.
find_package(CUDAToolkit REQUIRED)

add_library(ProjectCommon INTERFACE)

target_include_directories(ProjectCommon INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/utils/inc
)

target_link_libraries(ProjectCommon INTERFACE
    CUDA::cublas      # Represents libcublas
    CUDA::cusolver    # Represents libcusolver
    CUDA::curand      # Represents libcurand
)

# nvtx target name varies by CUDA version. Prefer nvtx3 if available.
if (TARGET CUDA::nvtx3)
    target_link_libraries(ProjectCommon INTERFACE CUDA::nvtx3)
elseif (TARGET CUDA::nvToolsExt)
    target_link_libraries(ProjectCommon INTERFACE CUDA::nvToolsExt)
endif()

# default release build
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
endif ()

option(USE_MGS "Enable USE_MGS macro" OFF)
if(USE_MGS)
    # If USE_MGS is ON, add the definition to `ProjectCommon`, which propagates to all linked targets.
    target_compile_definitions(ProjectCommon INTERFACE USE_MGS)
endif()

set(TSQR_SRCS
    src/TSQR.cu
    src/ReconstructWY.cu
    utils/myutils.cu
)

set(TSQR_FLOAT_SRCS
    src/TSQR_float.cu
    src/ReconstructWY_float.cu
    utils/myutils.cu
)

add_library(TSQR STATIC ${TSQR_SRCS})
# Link it to ProjectCommon to inherit shared includes, compile options, and link libraries.
target_link_libraries(TSQR PRIVATE ProjectCommon)
# Create libTSQR_float.a
add_library(TSQR_float STATIC ${TSQR_FLOAT_SRCS})
target_link_libraries(TSQR_float PRIVATE ProjectCommon)

set(GEQRF_CHECK_SINGLE_SRCS
    src/geqrf_check_single.cu
    utils/myutils.cu
)

set(GEQRF_SINGLE_SRCS
    src/geqrf_single.cu
    utils/myutils.cu
)

add_executable(geqrf_check_single ${GEQRF_CHECK_SINGLE_SRCS})
target_link_libraries(geqrf_check_single PRIVATE ProjectCommon)

add_executable(geqrf_single ${GEQRF_SINGLE_SRCS})
target_link_libraries(geqrf_single PRIVATE ProjectCommon)

# GEMMUL8 兼容
set(GEQRF_CHECK_SINGLE_GEMMUL8_SRCS
    src/geqrf_check_single_gemmul8.cu
    utils/myutils.cu
)
add_executable(geqrf_check_single_gemmul8 ${GEQRF_CHECK_SINGLE_GEMMUL8_SRCS})
target_link_libraries(geqrf_check_single_gemmul8 PRIVATE ProjectCommon gemmul8)
target_include_directories(geqrf_check_single_gemmul8 PUBLIC ${GEMMUL8_INCLUDE_DIR})

set(GEQRF_CHECK_DOUBLE_SRCS
    src/geqrf_check_double.cu
    utils/myutils.cu
)
add_executable(geqrf_check_double ${GEQRF_CHECK_DOUBLE_SRCS})
target_link_libraries(geqrf_check_double PRIVATE ProjectCommon)

set(GEQRF_CHECK_DOUBLE_GEMMUL8_SRCS
    src/geqrf_check_double_gemmul8.cu
    utils/myutils.cu
)
add_executable(geqrf_check_double_gemmul8 ${GEQRF_CHECK_DOUBLE_GEMMUL8_SRCS})
target_link_libraries(geqrf_check_double_gemmul8 PRIVATE ProjectCommon gemmul8)
target_include_directories(geqrf_check_double_gemmul8 PUBLIC ${GEMMUL8_INCLUDE_DIR})

set(GEQRF_DOUBLE_SRCS
    src/geqrf_double.cu
    utils/myutils.cu
)
add_executable(geqrf_double ${GEQRF_DOUBLE_SRCS})
target_link_libraries(geqrf_double PRIVATE ProjectCommon)

set(GEQRF_DOUBLE_GEMMUL8_SRCS
    src/geqrf_double_gemmul8.cu
    utils/myutils.cu
)
add_executable(geqrf_double_gemmul8 ${GEQRF_DOUBLE_GEMMUL8_SRCS})
target_link_libraries(geqrf_double_gemmul8 PRIVATE ProjectCommon gemmul8)
target_include_directories(geqrf_double_gemmul8 PUBLIC ${GEMMUL8_INCLUDE_DIR})

set(GEQRF_SINGLE_GEMMUL8_SRCS
    src/geqrf_single_gemmul8.cu
    utils/myutils.cu
)
add_executable(geqrf_single_gemmul8 ${GEQRF_SINGLE_GEMMUL8_SRCS})
target_link_libraries(geqrf_single_gemmul8 PRIVATE ProjectCommon gemmul8)
target_include_directories(geqrf_single_gemmul8 PUBLIC ${GEMMUL8_INCLUDE_DIR})
