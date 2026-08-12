# Standalone build (NOT part of the ROS2 ament package / colcon build) for the
# from-scratch dufomap Python binding in src/bind.cpp. See that file's header comment
# for why this exists instead of `pip install dufomap`.
#
# Build:
#   pip install nanobind          # provides the CMake package this file finds below
#                                  # (must be <=2.6.0 on Python 3.8 -- 2.14.0 dropped
#                                  # 3.8 support; irrelevant on newer Python)
#   sudo apt install liblz4-dev liblzf-dev   # REQUIRED, not optional: UFO's
#                                             # include/ufo/map/point_cloud.hpp
#                                             # unconditionally #includes <liblzf/lzf.h>
#                                             # (used by its PCD-compressed-binary
#                                             # reader/writer, which our bind.cpp never
#                                             # calls, but the header still requires the
#                                             # dependency to compile at all)
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j$(nproc)
#   # then either: cp build/dufomap_bind*.so <your venv>/lib/pythonX.Y/site-packages/
#   # or: add build/ to PYTHONPATH
#   # and: cp -r python_pkg/dufomap <same site-packages dir>  (the pure-Python glue --
#   # dufomap.utils.pose_check() etc -- copied verbatim from the real PyPI package,
#   # unmodified, since it has no compiled dependency and isn't platform-specific)
#
# On Jetson/aarch64 specifically: needs g++-10 (<concepts> isn't in GCC 9's libstdc++,
# even with -std=c++20) -- pass -DCMAKE_CXX_COMPILER=g++-10. GCC 9 is Jetson's default.

cmake_minimum_required(VERSION 3.16)
project(dufomap_custom_bind CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Vendored, patched copy of Kin-Zhang/dufomap's include/ufo -- see vendor/README.md
# for exactly what's patched and why (one ARM portability fix, one real correctness
# bug that silently broke every negative-coordinate point on aarch64). Overridable via
# -DUFO_INCLUDE_DIR if you need to point at a different/newer copy, but re-read
# vendor/README.md's "Updating this vendor copy" section first if you do -- the
# correctness bug in particular has no compiler warning and no crash, it just silently
# produces wrong detections.
if(NOT DEFINED UFO_INCLUDE_DIR)
  set(UFO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/ufo_include")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)
pkg_check_modules(LZF REQUIRED liblzf)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE NB_DIR
  RESULT_VARIABLE NB_DIR_RESULT)
if(NOT NB_DIR_RESULT EQUAL 0)
  message(FATAL_ERROR "`python -m nanobind --cmake_dir` failed -- is nanobind "
                       "installed in this Python environment? (`pip install nanobind`)")
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# TBB / UFO_PARALLEL: the official package's own startup log
# ("Set num_threads: 16") shows it runs the multi-threaded std::execution::par_unseq
# path in include/ufo/map/integration/integration.hpp (gated on UFO_PARALLEL) with
# num_threads=0 meaning "use all available", not "single-threaded". Originally left
# this disabled here to avoid an extra Jetson dependency; re-enabled after an
# x86 same-bag comparison against the official package (84.7% exact per-frame match,
# corr=0.9949 without this) to test whether matching the parallel code path narrows
# the remaining gap further (voxel-boundary points are sensitive to floating-point
# summation order, which differs between parallel and sequential execution).
find_package(TBB QUIET)
if(TBB_FOUND)
  message(STATUS "TBB found, enabling UFO_PARALLEL")
else()
  message(STATUS "TBB not found, UFO_PARALLEL disabled (single-threaded fallback)")
endif()

find_package(OpenMP)

nanobind_add_module(dufomap_bind NB_STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/bind.cpp)
target_include_directories(dufomap_bind PRIVATE ${UFO_INCLUDE_DIR} ${LZ4_INCLUDE_DIRS} ${LZF_INCLUDE_DIRS})
target_link_directories(dufomap_bind PRIVATE ${LZ4_LIBRARY_DIRS} ${LZF_LIBRARY_DIRS})
target_link_libraries(dufomap_bind PRIVATE ${LZ4_LIBRARIES} ${LZF_LIBRARIES})
target_compile_options(dufomap_bind PRIVATE -O3 -DNDEBUG)
if(OpenMP_CXX_FOUND)
  target_link_libraries(dufomap_bind PRIVATE OpenMP::OpenMP_CXX)
endif()
if(TBB_FOUND)
  target_compile_definitions(dufomap_bind PRIVATE UFO_PARALLEL=1)
  target_link_libraries(dufomap_bind PRIVATE TBB::tbb)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=cortex-a78ae" COMPILER_SUPPORTS_CORTEX_A78AE)
  if(COMPILER_SUPPORTS_CORTEX_A78AE)
    target_compile_options(dufomap_bind PRIVATE -mcpu=cortex-a78ae)
  endif()
endif()
