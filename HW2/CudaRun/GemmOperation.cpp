#include "GemmOperation.h"

#include <iostream>

#include "util.h"

template class GemmOperation<float>;

template <class T>
void GemmOperation<T>::AllocateHost()
{
    h_A = (T*)malloc((size_t)widthA * (size_t)heightA * sizeof(T));
    h_B = (T*)malloc((size_t)widthB * (size_t)heightB * sizeof(T));
    h_C = (T*)malloc((size_t)widthB * (size_t)heightA * sizeof(T));

    if (h_A == nullptr || h_B == nullptr || h_C == nullptr)
    {
        std::cerr << "failed to allocate host vectors" << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <class T>
void GemmOperation<T>::InitData()
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

template <class T>
void GemmOperation<T>::VerifyResult()
{
    T* c = new float[(size_t)widthB * (size_t)heightA];

    T buf = 0.0;

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
                T a = h_A[i + rowC * widthA];
                T b = h_B[colC + i * widthB];
                T prod = a * b;

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

template <class T>
void GemmOperation<T>::FreeHost()
{
    free(h_A);
    free(h_B);
    free(h_C);
}
