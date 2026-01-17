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


H="external/RAMA/include/rama_cuda.h"
CU="external/RAMA/src/rama_cuda.cu"
PATCH_H="rama_cuda_patch.h"
PATCH_CU="rama_cuda_patch.cu"

need_patch=0

if [[ ! -f "$H" || ! -f "$CU" ]]; then
  echo "ERROR: $H or $CU not found in current directory."
  exit 1
fi
if [[ ! -f "$PATCH_H" || ! -f "$PATCH_CU" ]]; then
  echo "ERROR: $PATCH_H or $PATCH_CU not found in current directory."
  exit 1
fi

# Check for the overload declaration/definition (match by function name + key params)
if ! grep -Eq 'rama_cuda\s*\(\s*const\s+thrust::device_vector<int>\s*&\s*i\s*,\s*const\s+thrust::device_vector<int>\s*&\s*j\s*,\s*thrust::device_vector<float>\s*&&\s*costs\s*,\s*const\s+multicut_solver_options\s*&\s*opts\s*,\s*int\s+device\s*\)' "$H" "$CU"; then
  need_patch=1
fi

# Check for the batched function declaration/definition
if ! grep -Eq 'rama_cuda_batched\s*\(\s*const\s+thrust::device_vector<int>\s*&\s*i\s*,\s*const\s+thrust::device_vector<int>\s*&\s*j\s*,\s*const\s+thrust::device_vector<float>\s*&\s*costs_be\s*,\s*int\s+B\s*,\s*int\s+E\s*,\s*int\s+num_nodes\s*,\s*const\s+multicut_solver_options\s*&\s*opts\s*,\s*int\s+device\s*\)' "$H" "$CU"; then
  need_patch=1
fi

if [[ "$need_patch" -eq 1 ]]; then
    echo "MISSING: One or more signatures not found. Appending patches..."

    # Append patches, separated by a newline marker
    {
    echo
    echo "// ---- appended from ${PATCH_H} on $(date -Iseconds) ----"
    cat "$PATCH_H"
    } >> "$H"

    {
    echo
    echo "// ---- appended from ${PATCH_CU} on $(date -Iseconds) ----"
    cat "$PATCH_CU"
    } >> "$CU"

    echo "DONE: Appended $PATCH_H -> $H and $PATCH_CU -> $CU"
fi


# setup without cuda: cmake -B build -S . -DENABLE_CUDA=OFF
cmake -B build -S .
