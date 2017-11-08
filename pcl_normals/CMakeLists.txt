cmake_minimum_required(VERSION 3.7)
project(Normals)

#project includes
include_directories(inc)

#project sources
file(GLOB SOURCES "src/*.cpp" "src/*.c")
add_executable(normals ${SOURCES})

#PCL
find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
target_link_libraries(normals ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES})
