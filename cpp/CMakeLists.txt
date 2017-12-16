cmake_minimum_required(VERSION 3.7)
project(PlaneDetection)

#project includes
include_directories(inc)

#project sources
file(GLOB SOURCES "src/*.cpp" "src/*.c")
add_executable(plane_detection ${SOURCES})

