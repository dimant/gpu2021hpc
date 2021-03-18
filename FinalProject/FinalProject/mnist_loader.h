#pragma once

#include <string>

struct data_dim
{
    int n;
    int rows;
    int cols;
};

// Read MNIST dataset images into contiguous float array and normalize values to range (0,1)
// Caller must delete[] returned float array
float* load_images(std::string path, data_dim& dd);

// Read MNIST dataset labels as one-hot-encoded arrays, placed into contiguous float array
// Caller must delete[] returned float array
float* load_labels(std::string path, size_t& labels);

