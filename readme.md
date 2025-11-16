# Image Compression with Multicut

## Setup and Dependencies
Download the benchmark dataset from [https://qoiformat.org/benchmark/](https://qoiformat.org/benchmark/).
Put the contents in a new directory called `/dataset`.

#### Libraries
 - `opencv` (4.12.0)
 - `libtorch` (2.9.1)

## Build
```
mkdir build
cd build
cmake ..
cmake --build .
```

## Execute
```
cd build
./image_compression
```
