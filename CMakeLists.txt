cmake_minimum_required(VERSION 3.16)

####################
# CUDA switch
####################
option(ENABLE_CUDA "Enable CUDA and RAMA" ON)

if(ENABLE_CUDA)
    project(image_compression LANGUAGES CXX CUDA)
else()
    project(image_compression LANGUAGES CXX)
endif()

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

include_directories(${OpenCV_INCLUDE_DIRS})

####################
# RAMA only when CUDA enabled
####################
if(ENABLE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 89)
    add_subdirectory(external/RAMA)
endif()

####################
# compress
####################
if(ENABLE_CUDA)
add_executable(compress
    src/compress.cpp
    src/metadata.cpp
    src/image_loader.cpp
    src/image_writer.cpp
    src/image_slicer.cpp
    src/rama_wrapper.cu
)
target_link_libraries(compress PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    rama_cuda
    RAMA
)
else()
add_executable(compress
    src/compress.cpp
    src/metadata.cpp
    src/image_loader.cpp
    src/image_writer.cpp
    src/image_slicer.cpp
    src/rama_wrapper.cu
)
target_link_libraries(compress PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)
endif()

target_include_directories(compress PRIVATE
    include
    external/RAMA/include
)
set_target_properties(compress PROPERTIES
    CXX_STANDARD 17
    CUDA_ARCHITECTURES 89
)

####################
# reassemble
####################
add_executable(reassemble
    src/reassemble.cpp
    src/metadata.cpp
    src/image_loader.cpp
    src/image_writer.cpp
)
target_link_libraries(reassemble PRIVATE
    ${OpenCV_LIBS}
)
target_include_directories(reassemble PRIVATE
    include
)

####################
# image converter
####################
add_executable(image_converter
    src/image_converter.cpp
    src/image_loader.cpp
    src/image_writer.cpp
)
target_link_libraries(image_converter PRIVATE
    ${OpenCV_LIBS}
)
target_include_directories(image_converter PRIVATE
    include
)
target_compile_options(image_converter PRIVATE -O3)

####################
# pretraining
####################
add_executable(pretraining
    src/fcn/pretraining.cpp
    src/slic_edge.cpp
    src/canny_edge.cpp
    src/graph_based_edge.cpp
    src/watershed_edge.cpp
    src/image_loader.cpp
    src/image_writer.cpp
)
target_link_libraries(pretraining PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)
target_include_directories(pretraining PRIVATE
    include
)
target_compile_options(pretraining PRIVATE -O3)

####################
# training
####################
if(ENABLE_CUDA)
add_executable(training
    src/fcn/training.cpp
    src/rama_wrapper.cu
    src/slic_edge.cpp
    src/canny_edge.cpp
    src/graph_based_edge.cpp
    src/watershed_edge.cpp
    src/image_loader.cpp
    src/png_size_estimator.cu
    src/segment_stats.cu
    src/compute_rewards.cu
)
target_link_libraries(training PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    rama_cuda
    RAMA
)
else()
add_executable(training
    src/fcn/training.cpp
    src/slic_edge.cpp
    src/canny_edge.cpp
    src/graph_based_edge.cpp
    src/watershed_edge.cpp
)
target_link_libraries(training PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)
endif()
target_include_directories(training PRIVATE
    include
    external/RAMA/include
)
set_target_properties(training PROPERTIES
    CUDA_ARCHITECTURES 89
)
target_compile_options(training PRIVATE -O3)
