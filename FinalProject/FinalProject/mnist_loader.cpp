#pragma once

#include <string>
#include <fstream>

#include "mnist_loader.h"

int reverse_int(int i)
{
    unsigned char buf[4];

    buf[0] =         i & 255;
    buf[1] = (i >>  8) & 255;
    buf[2] = (i >> 16) & 255;
    buf[3] = (i >> 24) & 255;

    int result = 0;

    result  = buf[0] << 24;
    result += buf[1] << 16;
    result += buf[2] << 8;
    result += buf[3];

    return result;
}

int read_int(std::ifstream& istream)
{
    int result = 0;
    istream.read((char*)&result, sizeof(int));
    return reverse_int(result);
}

void consume_magic_number(std::ifstream& istream, int magic)
{
    int magic_number = read_int(istream);

    if (magic_number != magic)
    {
        throw std::runtime_error("Invalid MNIST image file!");
    }
}

void read_data_dim(std::ifstream& istream, data_dim& dd)
{
    dd.n = read_int(istream);
    dd.rows = read_int(istream);
    dd.cols = read_int(istream);
}

void read_data(std::ifstream& istream, unsigned char** data, const data_dim dd)
{
    unsigned char* _data = new unsigned char[(size_t)dd.n * (size_t)dd.rows * (size_t)dd.cols];

    for (int i = 0; i < dd.n; i++)
    {
        istream.read((char*)&(_data)[i * dd.rows * dd.cols], (size_t)dd.rows * (size_t)dd.cols);
    }

    *data = _data;
}

float* load_images(std::string path, data_dim& dd)
{
    std::ifstream istream(path, std::ios::binary);

    if (!istream.is_open())
    {
        throw std::runtime_error("Cannot open file `" + path + "`!");
    }

    consume_magic_number(istream, 2051);

    read_data_dim(istream, dd);

    unsigned char* data;
    read_data(istream, &data, dd);

    size_t nn = (size_t)dd.n * (size_t)dd.rows * (size_t)dd.cols;
    float* result = new float[nn];

    for (size_t i = 0; i < nn; i++)
    {
        if (0 == data[i])
        {
            result[i] = 0.0f;
        }
        else
        {
            result[i] = 1.0f / (int)data[i];
        }
    }

    delete[] data;
    istream.close();

    return result;
}

float* load_labels(std::string path, size_t& labels)
{
    std::ifstream istream(path, std::ios::binary);

    if (!istream.is_open())
    {
        throw std::runtime_error("Cannot open file `" + path + "`!");
    }

    consume_magic_number(istream, 2049);

    int classes = 10;
    
    labels = read_int(istream);

    unsigned char* data = new unsigned char[(size_t)labels];
    istream.read((char*)data, labels);

    float* result = new float[(size_t)classes * (size_t)labels];

    for (int label = 0; label < labels; label++)
    {
        for (int klass = 0; klass < classes; klass++)
        {
            if (klass == (int)data[label])
            {
                result[label * classes + klass] = 1.0f;
            }
            else
            {
                result[label * classes + klass] = 0.0f;
            }
        }
    }

    delete[] data;
    istream.close();

    return result;
}
