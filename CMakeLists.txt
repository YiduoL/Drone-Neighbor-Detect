cmake_minimum_required(VERSION 3.5)
project(point_lio)

# Use C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )

# Default to Release if the caller (e.g. colcon) didn't specify a build type. -O3 below
# is applied unconditionally regardless of this, but CMAKE_BUILD_TYPE affects other
# things (e.g. some find_package config choices), so pin it explicitly rather than
# silently building with CMake's default empty build type.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Step-0 profiling instrumentation (include/prof_timing.h) -- off by default so a normal
# build has zero trace of it (not even a runtime branch). Enable with
# `colcon build --cmake-args -DPOINTLIO_PROFILE=ON` to get per-frame mean/p50/p95/p99
# timing summaries printed every 200 frames for: ekf_update, nn_search, map_incremental,
# deskew (see prof_timing.h and its call sites for exact scope boundaries).
option(POINTLIO_PROFILE "Enable lightweight timing instrumentation (see include/prof_timing.h)" OFF)
if(POINTLIO_PROFILE)
  add_compile_definitions(POINTLIO_PROFILE)
  message(STATUS "POINTLIO_PROFILE enabled -- timing instrumentation compiled in")
endif()

# Compiler settings for performance and threading
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")
# -DNDEBUG strips eigen_assert (and standard assert()) -- previously missing here, so
# every per-point Eigen op in the hot loop was paying for debug bounds/shape checks in
# what was otherwise an -O3 build. This is a pure compile-flag change, not a behavior
# change: any assert that would fire is a bug, not something relied upon.
add_compile_options(-O3 -DNDEBUG -pthread -fexceptions)

# On aarch64 (Jetson's Cortex-A78AE), target the specific core instead of the generic
# ARM baseline ISA the compiler defaults to. Probed via check_cxx_compiler_flag rather
# than assumed, since this must still build on non-Orin aarch64 / older GCC. Deliberately
# NOT -march=native: the binary may run on a different machine than it was built on.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)|(arm64)")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mcpu=cortex-a78ae" COMPILER_SUPPORTS_CORTEX_A78AE)
  if(COMPILER_SUPPORTS_CORTEX_A78AE)
    add_compile_options(-mcpu=cortex-a78ae)
    message(STATUS "aarch64: targeting -mcpu=cortex-a78ae")
  else()
    message(STATUS "aarch64: -mcpu=cortex-a78ae not supported by this compiler, staying with plain -O3 -DNDEBUG")
  endif()
endif()

# Processor count and definitions
message(STATUS "Current CPU architecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
  include(ProcessorCount)
  ProcessorCount(N)
  message(STATUS "Processor number:  ${N}")
  if(N GREATER 5)
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM=4)
    message(STATUS "core for MP:  3")
  elseif(N GREATER 3)
    math(EXPR PROC_NUM "${N} - 2")
    add_definitions(-DMP_EN)
    add_definitions(-DMP_PROC_NUM="${PROC_NUM}")
    message(STATUS "core for MP:  ${PROC_NUM}")
  else()
    add_definitions(-DMP_PROC_NUM=1)
  endif()
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

# Find packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(livox_ros_driver2 REQUIRED)

find_package(Eigen3 REQUIRED)


# OpenMP
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()

find_package(PythonLibs REQUIRED) # Consider using Pybind11 for ROS2
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")

include_directories(
        include
        ${EIGEN3_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIRS}
)

# Declare a ROS2 executable
add_executable(pointlio_mapping src/laserMapping.cpp include/ikd-Tree/ikd_Tree.cpp src/parameters.cpp src/preprocess.cpp src/Estimator.cpp)
ament_target_dependencies(pointlio_mapping
        rclcpp
        rclpy
        geometry_msgs
        nav_msgs
        sensor_msgs
        pcl_ros
        pcl_conversions
        tf2_ros
        visualization_msgs
        livox_ros_driver2
)
target_link_libraries(pointlio_mapping ${PYTHON_LIBRARIES})
target_include_directories(pointlio_mapping PRIVATE ${PYTHON_INCLUDE_DIRS})

# Install the executable
install(TARGETS
        pointlio_mapping
        DESTINATION lib/${PROJECT_NAME}
)

install(
        DIRECTORY config launch rviz_cfg
        DESTINATION share/${PROJECT_NAME}
)

# Export dependencies
ament_export_dependencies(rclcpp
  rclpy
  geometry_msgs
  nav_msgs
  sensor_msgs
  pcl_ros
  pcl_conversions
  tf2_ros
  visualization_msgs
  livox_ros_driver2
  Eigen3
)

ament_package()
