cmake_minimum_required(VERSION 3.20)

project(bmo_jetson LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(BMO_TARGET_JETSON "Target NVIDIA Jetson Orin Nano (sm_87)" ON)
option(BMO_ENABLE_CUDA "Enable CUDA backend" ON)
option(BMO_USE_FETCHCONTENT "Fetch llama.cpp / ggml via Git if not present locally" ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "87" CACHE STRING "CUDA Architecture (87 for Jetson Orin Nano)" FORCE)
endif()

message(STATUS "[BMO Jetson Build] Target CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Compiler definitions for Jetson Orin Nano
add_compile_definitions(
    BMO_ENABLE_CUDA=1
    BMO_TARGET_JETSON=1
    BMO_JETSON=1
    GGML_USE_CUDA=1
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
    add_compile_options(
        "$<$<COMPILE_LANGUAGE:C,CXX>:-O3>"
        "$<$<COMPILE_LANGUAGE:C,CXX>:-march=armv8.2-a>"
    )
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/llama.cpp" CACHE PATH "Path to llama.cpp source tree")

if(EXISTS "${LLAMA_CPP_DIR}/ggml/CMakeLists.txt")
    message(STATUS "[BMO Jetson Build] Using local llama.cpp/ggml at: ${LLAMA_CPP_DIR}")
    set(GGML_CUDA ON CACHE BOOL "Enable ggml CUDA backend" FORCE)
    add_subdirectory("${LLAMA_CPP_DIR}/ggml" "${CMAKE_BINARY_DIR}/ggml-build")
    set(GGML_INCLUDE_DIRS "${LLAMA_CPP_DIR}/ggml/include")
elseif(BMO_USE_FETCHCONTENT)
    include(FetchContent)
    message(STATUS "[BMO Jetson Build] Fetching upstream ggml via FetchContent...")
    set(GGML_CUDA ON CACHE BOOL "Enable ggml CUDA backend" FORCE)
    FetchContent_Declare(
        llamacpp
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
        GIT_TAG        master
    )
    FetchContent_MakeAvailable(llamacpp)
    set(GGML_INCLUDE_DIRS "${llamacpp_SOURCE_DIR}/ggml/include")
else()
    message(FATAL_ERROR "Could not find llama.cpp at ${LLAMA_CPP_DIR} and BMO_USE_FETCHCONTENT is OFF.")
endif()

# Discover ggml targets
set(_ggml_targets ggml ggml-base ggml-cpu ggml-cuda)
set(_link_targets "")
foreach(t IN LISTS _ggml_targets)
    if(TARGET ${t})
        list(APPEND _link_targets ${t})
    endif()
endforeach()

message(STATUS "[BMO Jetson Build] Discovered GGML targets: ${_link_targets}")

# Source files for BMO inference engine
set(BMO_SOURCES
    src/bmo.cpp
    src/bmo_compute.cpp
    src/bmo_cuda_kernels.cu
    src/bmo_cuda_kernels_proto.cu
)

# 1. Static library: libbmo.a
add_library(bmo STATIC ${BMO_SOURCES})
target_include_directories(bmo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" ${GGML_INCLUDE_DIRS})
target_link_libraries(bmo PRIVATE ${_link_targets} CUDA::cudart CUDA::cuda_driver Threads::Threads)
set_target_properties(bmo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# 2. Shared library: libbmo.so (for Python runtime / edge client bridge)
add_library(bmo_shared SHARED ${BMO_SOURCES} src/bmo_api.cpp)
set_target_properties(bmo_shared PROPERTIES
    OUTPUT_NAME bmo
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
)
target_include_directories(bmo_shared PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" ${GGML_INCLUDE_DIRS})
target_compile_definitions(bmo_shared PRIVATE BMO_BUILDING_SHARED=1)
target_link_libraries(bmo_shared PRIVATE ${_link_targets} CUDA::cudart CUDA::cuda_driver Threads::Threads)

# 3. Main inference & latency harness binary: bmo_engine
add_executable(bmo_engine src/main.cpp)
target_include_directories(bmo_engine PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${GGML_INCLUDE_DIRS})
target_link_libraries(bmo_engine PRIVATE bmo ${_link_targets} CUDA::cudart CUDA::cuda_driver Threads::Threads)

# 4. Standalone Q4_0 kernel bandwidth benchmark: bmo_kernel_bench
add_executable(bmo_kernel_bench src/bmo_kernel_bench.cu)
target_include_directories(bmo_kernel_bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(bmo_kernel_bench PRIVATE CUDA::cudart Threads::Threads)

# 5. Native C++ Real-Time Audio Daemon: bmo_daemon
add_executable(bmo_daemon src/bmo_daemon.cpp)
target_include_directories(bmo_daemon PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" ${GGML_INCLUDE_DIRS} /usr/local/cuda/include)
target_link_libraries(bmo_daemon PRIVATE bmo_shared ${_link_targets} CUDA::cudart CUDA::cuda_driver Threads::Threads nvinfer asound)
