#include "BlurOperation.h"

#include <iostream>

#include "util.h"

void BlurOperation::AllocateHost()
{
    h_A = (char*)malloc(cols * rows * sizeof(char));
    h_B = (char*)malloc(cols * rows * sizeof(char));

    if (h_A == nullptr || h_B == nullptr)
    {
        std::cerr << "failed to allocate host vectors" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void BlurOperation::InitData()
{
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            size_t idx = row * cols + col;
            h_A[idx] = (int)idx;
        }
    }
}

void BlurOperation::VerifyResult()
{
    char* b = new char[rows * cols];

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {

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

void BlurOperation::FreeHost()
{
    free(h_A);
    free(h_B);
}
