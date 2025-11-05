# Image Compression with Multicut

## Setup and Dependencies
Download the benchmark dataset from [https://qoiformat.org/benchmark/](https://qoiformat.org/benchmark/).
Put the contents in a new directory called `/dataset`.

### libpng
This project uses `libpng`. To install on Linux:
```
sudo apt install libpng-dev
```
or on MacOS with `homebrew`:
```
brew install libpng
```

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
