#pragma once

#include <string>

// Read MNIST dataset images into contiguous float array and normalize values to range (0,1)
// Caller must delete[] returned float array
float* load_images(std::string path);

// Read MNIST dataset labels as one-hot-encoded arrays, placed into contiguous float array
// Caller must delete[] returned float array
float* load_labels(std::string path);

