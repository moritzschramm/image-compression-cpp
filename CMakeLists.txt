cmake_minimum_required(VERSION 3.16)
project(image_compression LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PNG REQUIRED)

add_executable(image_compression
    src/main.cpp
    src/huffman.cpp
    src/image_loader.cpp
)

target_link_libraries(image_compression
    PNG::PNG
)

target_include_directories(image_compression PRIVATE include)
