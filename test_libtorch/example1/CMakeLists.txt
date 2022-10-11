cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(1minimal_example)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(1minimal_example 1minimal_example.cpp)
target_link_libraries(1minimal_example "${TORCH_LIBRARIES}")
set_property(TARGET 1minimal_example PROPERTY CXX_STANDARD 14)