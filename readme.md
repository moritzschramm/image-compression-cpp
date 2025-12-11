# Image Compression with Multicut

Use DL to predict weights for Multicut to segment image into well compressable image slices.
Position in original image is stored as well, so reassembly to original image is possible.

## Setup and Dependencies
Execute `./setup.sh`. (Setup without CUDA: `cmake -B build -S . -DENABLE_CUDA=OFF`; many features won't work)

Download the benchmark dataset from [https://qoiformat.org/benchmark/](https://qoiformat.org/benchmark/).
Put the contents in a new directory called `/dataset`.

#### Libraries
 - `opencv` (4.12.0)
 - `libtorch` (2.9.1)

## Build
NOTE: Right now the build only works with CUDA architecture 8.9 and CUDA Version 12.6. 
Change CMakeLists.txt for different versions. 
```
./build.sh
```

## Execute
```
./build/compress
./build/reassemble
./build/pretraining
./build/training
```

## Training

In pre-training, the network learns the edge costs for multicut segmentation based on local neighbor differences. 
The actual training uses the cumulative sum of the compressed slices' image size as a reward for online reinforcement learning.

## Multicut

This project uses [RAMA](https://github.com/pawelswoboda/RAMA) which solves the Multicut Problem on the GPU.
