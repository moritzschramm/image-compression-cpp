cmake_minimum_required(VERSION 3.16)
#project(image_compression LANGUAGES CXX CUDA)
project(image_compression LANGUAGES CXX)

#add_subdirectory(external/RAMA)

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
# compress
####################
add_executable(compress
    src/compress.cpp
    src/metadata.cpp
    src/image_loader.cpp
    src/image_writer.cpp
    src/image_slicer.cpp
    #src/rama_wrapper.cu
)
target_link_libraries(compress PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    #rama_cuda
    #RAMA
)
target_include_directories(compress PRIVATE
    include
    external/RAMA/include
)
set_property(TARGET compress PROPERTY CXX_STANDARD 17)

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
# pretraining
####################
add_executable(pretraining
    src/fcn/pretraining.cpp
    src/image_loader.cpp
    src/image_writer.cpp
    src/pattern_generator.cpp
)
target_link_libraries(pretraining PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
)
target_include_directories(pretraining PRIVATE
    include
)

####################
# training
####################
add_executable(training
    src/fcn/training.cpp
    #src/rama_wrapper.cu
)
target_link_libraries(training PRIVATE
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    #rama_cuda
    #RAMA
)
target_include_directories(training PRIVATE
    include
    external/RAMA/include
)

set_target_properties(compress  PROPERTIES CUDA_ARCHITECTURES 89)
set_target_properties(training PROPERTIES CUDA_ARCHITECTURES 89)
