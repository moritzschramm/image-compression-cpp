# Image Compression with Multicut [![build](https://github.com/moritzschramm/image-compression-cpp/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/moritzschramm/image-compression-cpp/actions/workflows/build.yml)


Use DL and RL to predict weights for Multicut to segment image into well compressable image slices.
Position in original image is stored as well, so reassembly to original image is possible.

## Setup and Dependencies
Execute `./setup.sh`. This creates the necessary directories, fetches and patches dependencies. 

Download the dataset from [ImageNet via kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data).
Put the contents of the images in a new directory called `/dataset`. The images can be converted to the correct format with the `image_converter` executable.

#### Libraries
 - `opencv` (4.12.0) (+ `ximgproc` module)
 - `libtorch` (2.9.1)

#### Configuration
See `include/configuration.h` for configuration options.

## Build
`NOTE:` Build only tested with CUDA architecture 8.9 and CUDA Version 12.6, but other versions may work as well. 
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

In pre-training, the network learns the edge costs for multicut segmentation based on a segmentation algorithm set in `configuration.h`. 
The actual training uses the cumulative sum of the compressed slices' image size as a reward for 
online reinforcement learning. 
For this, the costs for the multicut solver are sampled from probability distributions parametrized by the predicted network edge weights.

## Multicut

This project uses [RAMA](https://github.com/pawelswoboda/RAMA) which solves the Multicut Problem on the GPU.

## Future Work

To make the project work, there are a few improvments necessary:
 - Fix the oversegmentation of the pre-trained model. The recall of predicted "cut" edges looks good (>0.9), but the precision is bad (<0.25). This leads to oversegmentation which the subsequent RL training can't work with.
 - Implement actor-critic RL pattern. Right now, the RL training is a primitive stateless REINFORCE adaptation, which does not converge.
 - Use different image format. At the moment, PNG is used to encode the images. Since the whole pipeline runs on the GPU, a custom PNG file size estimator had to be generated. To encode images directly on the GPU, one could use `nvJPEG2k` (or `nvPNG`, if it is released by then).
