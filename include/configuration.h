#pragma once
#include <filesystem>

// change this so that the given directory is relative to the directory you are in while executing the program
const std::filesystem::path DATASET_DIR = "dataset/CLS-LOC/train";
const std::filesystem::path VAL_DATASET_DIR = "dataset/CLS-LOC/val";
const std::filesystem::path TEST_DATASET_DIR = "dataset/CLS-LOC/test";
const std::filesystem::path RESULTS_DIR = "./results";
const std::filesystem::path CACHE_DIR = "./.cache/imagecompression";
const std::string IMAGE_FORMAT = "png";
const int COMPRESSION_LEVEL = 4;

enum class EdgeTargetType {
    SLIC,
    CANNY,
    GRAPH,
    WATERSHED
};

const EdgeTargetType EDGE_TARGET = EdgeTargetType::GRAPH;
