cmake_minimum_required(VERSION 3.0.2)
project(vps_slam)

## Compile as C++14, supported in ROS Kinetic and newer
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -pthread")

# message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
# if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
#   include(ProcessorCount)
#   ProcessorCount(N)
#   message("Processer number:  ${N}")
#   if(N GREATER 5)
#     add_definitions(-DMP_EN)
#     add_definitions(-DMP_PROC_NUM=4)
#     message("core for MP:  4")
#   elseif(N GREATER 3)
#     math(EXPR PROC_NUM "${N} - 2")
#     add_definitions(-DMP_EN)
#     add_definitions(-DMP_PROC_NUM="${PROC_NUM}")
#     message("core for MP:  ${PROC_NUM}")
#   else()
#     add_definitions(-DMP_PROC_NUM=1)
#   endif()
# else()
#   add_definitions(-DMP_PROC_NUM=1)
# endif()

# find_package(OpenMP QUIET)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
find_package(Ceres REQUIRED)

message(STATUS "Using catkin version ${catkin_VERSION}")
message(STATUS "Using Ceres version ${CERES_VERSION}")

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  pcl_ros
  roscpp
  sensor_msgs
  visualization_msgs
  message_generation
  tf
)

add_message_files(
  FILES
  SurfelWithCovariance.msg
  MultiSurfels.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS geometry_msgs nav_msgs pcl_ros roscpp sensor_msgs visualization_msgs message_generation tf
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

link_directories(
  include
)

add_executable(PreProcessing src/PreProcessing.cpp)
target_link_libraries(PreProcessing ${catkin_LIBRARIES})

add_executable(SurfelExtraction src/SurfelExtraction.cpp)
target_link_libraries(SurfelExtraction ${catkin_LIBRARIES})

add_executable(SurfelMapping src/SurfelMapping.cpp)
target_link_libraries(SurfelMapping ${catkin_LIBRARIES} ${CERES_LIBRARIES})