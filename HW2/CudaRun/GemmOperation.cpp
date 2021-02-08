#include "GemmOperation.h"

#include <iostream>

#include "util.h"

void GemmOperation::AllocateHost()
{
    h_A = (float*)malloc((size_t)widthA * (size_t)heightA * sizeof(float));
    h_B = (float*)malloc((size_t)widthB * (size_t)heightB * sizeof(float));
    h_C = (float*)malloc((size_t)widthB * (size_t)heightA * sizeof(float));

    if (h_A == nullptr || h_B == nullptr || h_C == nullptr)
    {
        std::cerr << "failed to allocate host vectors" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void GemmOperation::InitData()
{
    for (int i = 0; i < heightA; i++)
    {
        for (int j = 0; j < widthA; j++)
        {
            h_A[i * widthA + j] = 1;
        }
    }

    for (int row = 0; row < heightB; row++)
    {
        for (int col = 0; col < widthB; col++)
        {
            h_B[row * widthB + col] = 1.0;
        }
    }
}

void GemmOperation::VerifyResult()
{
    float* c = new float[(size_t)widthB * (size_t)heightA];

    float buf = 0.0;

    int widthC = widthB;
    int heightC = heightA;

    // matrices are assumed to be square;
    for (int rowC = 0; rowC < heightC; rowC++)
    {
        for (int colC = 0; colC < widthC; colC++)
        {
            buf = 0.0;

            for (int i = 0; i < widthA; i++)
            {
                float a = h_A[i + rowC * widthA];
                float b = h_B[colC + i * widthB];
                float prod = a * b;

                buf += prod;
            }

            c[colC + rowC * widthC] = buf;
        }
    }

    double tolerance = DBL_MIN;

    for (int rowC = 0; rowC < heightC; rowC++)
    {
        for (int colC = 0; colC < widthC; colC++)
        {
            int idx = rowC * widthC + colC;
            double delta = fabs(h_C[idx] - c[idx]);

            if (tolerance < delta)
            {
                tolerance = delta;
            }
        }
    }

    checkTolerance(tolerance);

    delete[] c;
}

void GemmOperation::FreeHost()
{
    free(h_A);
    free(h_B);
    free(h_C);
}
