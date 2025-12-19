# Image Compression with Multicut

Use DL and RL to predict weights for Multicut to segment image into well compressable image slices.
Position in original image is stored as well, so reassembly to original image is possible.

## Setup and Dependencies
Execute `./setup.sh`. This creates the necessary directories, fetches and patches dependencies. 

Download the dataset from [https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data?select=ILSVRC](ImageNet via kaggle).
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

or build only single targets: `./build.sh <target>`, e.g. `./build.sh image_converter`

## Execute
```
./build/compress
./build/reassemble
./build/image_converter
./build/pretraining
./build/training
```

## Training

In pre-training, the network learns the edge costs for multicut segmentation based on local neighbor differences. 
The actual training uses the cumulative sum of the compressed slices' image size as a reward for online reinforcement learning.

## Multicut

This project uses [RAMA](https://github.com/pawelswoboda/RAMA) which solves the Multicut Problem on the GPU.
