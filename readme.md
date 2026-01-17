# Image Compression with Multicut

Use DL and RL to predict weights for Multicut to segment image into well compressable image slices.
Position in original image is stored as well, so reassembly to original image is possible.

## Setup and Dependencies
Execute `./setup.sh`. This creates the necessary directories, fetches and patches dependencies. 

Download the dataset from [ImageNet via kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data).
Put the contents of the images in a new directory called `/dataset`. The images can be converted to the correct format with the `image_converter` executable.

#### Libraries
 - `opencv` (4.12.0) (+ `ximgproc` module)
 - `libtorch` (2.9.1)

## Build
NOTE: Build only tested with CUDA architecture 8.9 and CUDA Version 12.6, but other versions may work as well. 
Change `CMakeLists.txt` to try different versions. 
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

In pre-training, the network learns the edge costs for multicut segmentation based on a superpixel partition. 
The actual training uses the cumulative sum of the compressed slices' image size as a reward for 
online reinforcement learning, by intepreting the edges as probabilities to sample from.

## Multicut

This project uses [RAMA](https://github.com/pawelswoboda/RAMA) which solves the Multicut Problem on the GPU.
