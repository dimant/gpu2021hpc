#include "DotProductOpenCL.h"

#include <iostream>

#include "ocl_util.h"

template class DotProductOpenCL<float>;
template class DotProductOpenCL<cl_float2>;
template class DotProductOpenCL<cl_float4>;

template <class T>
void DotProductOpenCL<T>::AllocateHost()
{
    DotProductOperation<T>::AllocateHost();
}

template <class T>
void DotProductOpenCL<T>::InitData()
{
    DotProductOperation<T>::InitData();
}

template <class T>
void DotProductOpenCL<T>::FreeHost()
{
    DotProductOperation<T>::FreeHost();
}

template <class T>
void DotProductOpenCL<T>::VerifyResult()
{
    DotProductOperation<T>::VerifyResult();
}

template <class T>
void DotProductOpenCL<T>::AllocateDevice()
{
    cl_int err;

    d_A = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, elements * sizeof(T), NULL, &err); clChkErr(err);
    d_B = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, elements * sizeof(T), NULL, &err); clChkErr(err);
    d_C = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, elements * sizeof(T), NULL, &err); clChkErr(err);
}

template <class T>
void DotProductOpenCL<T>::CopyToDevice()
{
    clChkErr(clEnqueueWriteBuffer(GetContext().queue, d_A, CL_TRUE, 0, elements * sizeof(T), h_A, 0, NULL, NULL));
    clChkErr(clEnqueueWriteBuffer(GetContext().queue, d_B, CL_TRUE, 0, elements * sizeof(T), h_B, 0, NULL, NULL));
}

template <class T>
void DotProductOpenCL<T>::Launch()
{

    cl_int err;

    cl_uint cl_elements = (cl_uint)elements;

    clChkErr(clSetKernelArg(GetContext().kernel, 0, sizeof(cl_mem), &d_A));
    clChkErr(clSetKernelArg(GetContext().kernel, 1, sizeof(cl_mem), &d_B));
    clChkErr(clSetKernelArg(GetContext().kernel, 2, sizeof(cl_mem), &d_C));
    clChkErr(clSetKernelArg(GetContext().kernel, 3, sizeof(cl_elements), &cl_elements));

    const size_t globalSize = localSize * numGroups;

    err = clEnqueueNDRangeKernel(
        GetContext().queue,
        GetContext().kernel,
        1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    clChkErr(clFinish(GetContext().queue));
}

template <class T>
void DotProductOpenCL<T>::CopyFromDevice()
{
    clChkErr(clEnqueueReadBuffer(GetContext().queue, d_C, CL_TRUE, 0, numGroups * sizeof(float), h_C, 0, NULL, NULL));

    for (int i = 1; i < numGroups; i++)
    {
        h_C[0] += h_C[i];
    }
}

template <class T>
void DotProductOpenCL<T>::FreeDevice()
{
    clChkErr(clReleaseMemObject(d_A));
    clChkErr(clReleaseMemObject(d_B));
    clChkErr(clReleaseMemObject(d_C));
}
