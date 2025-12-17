mkdir dataset
mkdir results
git submodule update --init --recursive

# custom patch to make RAMA compile
# maybe adding set(CMAKE_CUDA_ARCHITECTURES 89) is also necessary in CMakeLists.txt of RAMA
sed -i 's/include_directories(external/ECL-CC)/include_directories(${CMAKE_CURRENT_SOURCE_DIR}external/ECL-CC)/g' external/RAMA/CMakeLists.txt
sed -i 's/include_directories(include)/include_directories(${CMAKE_CURRENT_SOURCE_DIR}include)/g' external/RAMA/CMakeLists.txt
sed -i '/enable_testing()/d' external/RAMA/CMakeLists.txt
sed -i '/add_subdirectory(test)/d' external/RAMA/CMakeLists.txt
sed -i '/pybind11_add_module(rama_py rama_py.cu)/d' external/RAMA/src/CMakeLists.txt
sed -i '/target_link_libraries(rama_py PRIVATE multicut_text_parser rama_cuda RAMA)/d' external/RAMA/src/CMakeLists.txt

# setup without cuda: cmake -B build -S . -DENABLE_CUDA=OFF
cmake -B build -S .
