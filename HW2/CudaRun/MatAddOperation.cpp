#include <iostream>

#include "MatAddOperation.h"
#include "util.h"

void MatAddOperation::AllocateHost()
{
    size_t size = GetSize();

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        std::cerr << "failed to allocate host vectors" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void MatAddOperation::InitData()
{
    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            h_A[i * ncols + j] = float(i);
            h_B[i * ncols + j] = float(j);
        }
    }
}

void MatAddOperation::VerifyResult()
{
    double tolerance = DBL_MIN;

    for (int i = 0; i < nrows; i++)
    {
        for (int j = 0; j < ncols; j++)
        {
            double a = h_A[i * ncols + j];
            double b = h_B[i * ncols + j];
            double c = h_C[i * ncols + j];

            double delta = fabs(a + b - c);
            if (tolerance < delta)
            {
                tolerance = delta;
            }
        }
    }

    checkTolerance(tolerance);
}

void MatAddOperation::FreeHost()
{
    free(h_A);
    free(h_B);
    free(h_C);
}
