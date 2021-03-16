#pragma once
// MNIST dataset loading and pre-processing
//
// full_path strings must be fully-qualified path to a valid MNIST datafile of IDX file format 
//
#include <string>
#include <fstream>

typedef unsigned char uchar;

using namespace std;

auto reverseInt = [](int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
};

// Read MNIST dataset images into contiguous float array and normalize values to range (0,1)
// Caller must free( ) returned float array
void load_and_preproc_mnist_images(string full_path, float** _dataset, int& number_of_images, int& n_rows, int& n_cols) {

    ifstream file(full_path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0;

        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);

        uchar* _tmp_data = (uchar*)malloc(number_of_images * n_rows * n_cols * sizeof(uchar));
        float* _f32_data = NULL;
        CURT_CHK(cudaMallocManaged(&_f32_data,(number_of_images * n_rows * n_cols * sizeof(float))));

        //uchar** _dataset = new uchar * [number_of_images];
        for (int i = 0; i < number_of_images; i++) {
            file.read((char*)&_tmp_data[i * n_rows * n_cols], n_rows * n_cols);
        }

        // convert to float and normalize
        int total_vals = number_of_images * n_rows * n_cols;
        for (int i = 0; i < total_vals; ++i)
        {

            if (0 == (int)_tmp_data[i])
                _f32_data[i] = 0.0f;
            else
            {
                _f32_data[i] = 1.0f / (int)_tmp_data[i];
            }
        }
        free(_tmp_data);

        *_dataset = _f32_data;

    }
    else {
        throw runtime_error("Cannot open file `" + full_path + "`!");
    }
}

// Read MNIST dataset labels as one-hot-encoded arrays, placed into contiguous float array
// Caller must free( ) returned float array
void load_preproc_mnist_labels(string full_path, float** _dataset, int& number_of_labels) {

    ifstream file(full_path, ios::binary);

    if (file.is_open()) {
        int magic_number = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);

        if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

        file.read((char*)&number_of_labels, sizeof(number_of_labels));
        number_of_labels = reverseInt(number_of_labels);

        int number_of_MNIST_classes = 10;
        uchar* _tmp_data = (uchar*)malloc(number_of_labels * sizeof(uchar));
        float* _f32_data = NULL;
        CURT_CHK(cudaMallocManaged(&_f32_data, (number_of_labels * number_of_MNIST_classes * sizeof(float))));

        for (int i = 0; i < number_of_labels; i++) {
            file.read((char*)&_tmp_data[i], 1);
        }

        // convert to float and one-hot-encoding
        for (int i = 0; i < number_of_labels; ++i)
        {
            for (int j = 0; j < number_of_MNIST_classes; ++j)
            {
                if (j == (int)_tmp_data[i])
                    _f32_data[i * number_of_MNIST_classes + j] = 1.0f;
                else
                    _f32_data[i * number_of_MNIST_classes + j] = 0.0f;
            }

        }
        free(_tmp_data);

        *_dataset = _f32_data;


    }
    else {
        throw runtime_error("Unable to open file `" + full_path + "`!");
    }
}

