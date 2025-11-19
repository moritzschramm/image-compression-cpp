# Image Compression with Multicut

## Setup and Dependencies
Download the benchmark dataset from [https://qoiformat.org/benchmark/](https://qoiformat.org/benchmark/).
Put the contents in a new directory called `/dataset`.

Create a directory called `/results`.

#### Libraries
 - `opencv` (4.12.0)
 - `libtorch` (2.9.1)

## Build
```
mkdir build
cmake -B build -S .
cmake --build build -j
```

## Execute
```
./build/image_compression
```
