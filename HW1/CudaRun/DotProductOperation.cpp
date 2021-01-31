#include "DotProductOperation.h"

#include <iostream>
#include <math.h>

#include <device_launch_parameters.h>

template class DotProductOperation<float>;
template class DotProductOperation<float2>;
template class DotProductOperation<float4>;

template <class T>
void DotProductOperation<T>::AllocateHost()
{
    h_A = (T*)malloc(elements * sizeof(T));
    h_B = (T*)malloc(elements * sizeof(T));
    h_C = (float*)malloc(outputSize * sizeof(float));

    if (h_C != nullptr)
    {
        h_C = (float*)memset(h_C, 0, outputSize * sizeof(float));
    }

    if (h_A == nullptr || h_B == nullptr || h_C == nullptr)
    {
        std::cerr << "failed to allocate host vectors" << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <>
void DotProductOperation<float>::InitData()
{
    for (int i = 0; i < elements; i++)
    {
        h_A[i] = 1.0f;
        h_B[i] = log(float(i));
    }
}

template<>
void DotProductOperation<float2>::InitData()
{
    for (int i = 0; i < elements; i++)
    {
        h_A[i].x = 1.0f;
        h_A[i].y = 1.0f;
        h_B[i].x = log(float(i));
        h_B[i].y = log(float(i));
    }
}

template<>
void DotProductOperation<float4>::InitData()
{
    for (int i = 0; i < elements; i++)
    {
        h_A[i].x = 1.0f;
        h_A[i].y = 1.0f;
        h_A[i].z = 1.0f;
        h_A[i].w = 1.0f;
        h_B[i].x = log(float(i));
        h_B[i].y = log(float(i));
        h_B[i].z = log(float(i));
        h_B[i].w = log(float(i));
    }
}

template <>
void DotProductOperation<float>::VerifyResult()
{
    float c = 0.0;

    for (int i = 0; i < elements; i++)
    {
        c += h_A[i] * h_B[i];
    }

    if (fabs(h_C[0] - c) > 1e-5)
    {
        fprintf(stderr, "Result verification failed! %f != %f\n", c, h_C[0]);
        exit(EXIT_FAILURE);
    }
}

template <>
void DotProductOperation<float2>::VerifyResult()
{
    float c = 0.0;

    for (int i = 0; i < elements; i++)
    {
        c += h_A[i].x * h_B[i].x;
        c += h_A[i].y * h_B[i].y;
    }

    if (fabs(h_C[0] - c) > 1e-5)
    {
        fprintf(stderr, "Result verification failed!\n");
        exit(EXIT_FAILURE);
    }
}

template <>
void DotProductOperation<float4>::VerifyResult()
{
    float c = 0.0;

    for (int i = 0; i < elements; i++)
    {
        c += h_A[i].x * h_B[i].x;
        c += h_A[i].y * h_B[i].y;
        c += h_A[i].z * h_B[i].z;
        c += h_A[i].w * h_B[i].w;
    }

    if (fabs(h_C[0] - c) > 1e-5)
    {
        fprintf(stderr, "Result verification failed!\n");
        exit(EXIT_FAILURE);
    }
}

template <class T>
void DotProductOperation<T>::FreeHost()
{
    free(h_A);
    free(h_B);
    free(h_C);
}
