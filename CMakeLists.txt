cmake_minimum_required(VERSION 3.20)
project(fm_nsga2 VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Compiler flags ---
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -O2)
elseif(MSVC)
  add_compile_options(/W3 /O2)
endif()

# --- OpenMP ---
find_package(OpenMP REQUIRED)

# --- Include directories ---
include_directories(include baselines)

# --- Core library (shared between main and benchmark) ---
set(CORE_SOURCES
    src/dataset.cpp
    src/manifold.cpp
    src/nsga2.cpp
)

add_library(fm_nsga2_core STATIC ${CORE_SOURCES})
target_include_directories(fm_nsga2_core PUBLIC include baselines)
target_link_libraries(fm_nsga2_core PUBLIC OpenMP::OpenMP_CXX)

# --- Main executable (demo) ---
add_executable(fm_nsga2 src/main.cpp)
target_link_libraries(fm_nsga2 fm_nsga2_core)

# --- MGDA executable (gradient-based multi-objective) ---
add_executable(mgda src/mgda_main.cpp)
target_link_libraries(mgda fm_nsga2_core)

# --- Benchmark executable ---
add_executable(benchmark experiments/benchmark.cpp)
target_link_libraries(benchmark fm_nsga2_core)

# --- Copy data scripts to build dir ---
configure_file(
  ${CMAKE_SOURCE_DIR}/data/generate_datasets.py
  ${CMAKE_BINARY_DIR}/generate_datasets.py
  COPYONLY
)
configure_file(
  ${CMAKE_SOURCE_DIR}/experiments/visualize.py
  ${CMAKE_BINARY_DIR}/visualize.py
  COPYONLY
)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
