cmake_minimum_required(VERSION 3.16)
project(image_compression LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# libtorch setup
set(TORCH_DIR $ENV{TORCH_DIR})
if (NOT TORCH_DIR)
    message(FATAL_ERROR "Environment variable TORCH_DIR not set. Please export TORCH_DIR=/path/to/libtorch")
endif()
set(CMAKE_PREFIX_PATH "${TORCH_DIR}")
message(STATUS "Using TORCH_DIR = ${TORCH_DIR}")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(image_compression
    src/main.cpp
    src/image_loader.cpp
    src/image_slicer.cpp
)

target_link_libraries(image_compression PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)

target_include_directories(image_compression PRIVATE include)

set_property(TARGET image_compression PROPERTY CXX_STANDARD 17)
