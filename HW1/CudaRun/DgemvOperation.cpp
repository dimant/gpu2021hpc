#include "DgemvOperation.h"

#include <iostream>

#include <device_launch_parameters.h>
#include "cuda_util.h"

void DgemvOperation::AllocateHost()
{
    h_A = (double*)malloc(cols * rows * sizeof(double));
    h_x = (double*)malloc(cols * sizeof(double));
    h_y = (double*)malloc(rows * sizeof(double));
    h_z = (double*)malloc(rows * sizeof(double));

    if (h_A == nullptr || h_x == nullptr || h_y == nullptr || h_z == nullptr)
    {
        std::cerr << "failed to allocate host vectors" << std::endl;
        exit(EXIT_FAILURE);
    }

    h_z = (double*)memset(h_z, 0, rows * sizeof(double));
}

void DgemvOperation::InitData()
{
    alpha = 2.0;
    beta = 4.0;

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            size_t idx = row * cols + col;
            h_A[idx] = double(idx);
        }
    }

    for (int i = 0; i < cols; i++)
    {
        h_x[i] = i;
    }

    for (int i = 0; i < rows; i++)
    {
        h_y[i] = i;
    }
}

void DgemvOperation::VerifyResult()
{
    double* z = new double[rows];

    double buf = 0.0;

    for (int row = 0; row < rows; row++)
    {
        buf = 0.0f;

        for (int col = 0; col < cols; col++)
        {
            buf += alpha * h_A[row * cols + col] * h_x[col];
        }

        z[row] = buf + beta * h_y[row];
    }

    for (int i = 0; i < rows; i++)
    {
        if (fabs(h_z[i] - z[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed!\n");
            exit(EXIT_FAILURE);
        }
    }

    delete[] z;
}

void DgemvOperation::FreeHost()
{
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_z);
}
