#include "TransposeOperation.h"

#include <iostream>

#include "util.h"

void TransposeOperation::AllocateHost()
{
    h_A = (int*)malloc(cols * rows * sizeof(int));
    h_B = (int*)malloc(cols * rows * sizeof(int));

    if (h_A == nullptr || h_B == nullptr)
    {
        std::cerr << "failed to allocate host vectors" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void TransposeOperation::InitData()
{
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            size_t idx = row * cols + col;
            h_A[idx] = idx;
        }
    }
}

void TransposeOperation::VerifyResult()
{
    int* b = new int[rows * cols];

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            int index = row * rows + col;
            int transposedIndex = col * rows + row;

            b[index] = h_A[transposedIndex];
        }
    }

    for (int i = 0; i < rows * cols; i++)
    {
        if (h_B[i] != b[i])
        {
            std::cout << "Failed verifying transpose." << std::endl;
            break;
        }
    }

    delete[] b;
}

void TransposeOperation::FreeHost()
{
    free(h_A);
    free(h_B);
}
