#include "GemmOpenCL.h"

#include <iostream>

#include "ocl_util.h"

template class GemmOpenCL<float>;

template <class T>
void GemmOpenCL<T>::AllocateDevice()
{
    cl_int err;

    d_A = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, (size_t)widthA * (size_t)heightA * sizeof(T), NULL, &err); clChkErr(err);
    d_B = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, (size_t)widthB * (size_t)heightB * sizeof(T), NULL, &err); clChkErr(err);
    d_C = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, (size_t)widthB * (size_t)heightA * sizeof(T), NULL, &err); clChkErr(err);
}

template <class T>
void GemmOpenCL<T>::CopyToDevice()
{
    clChkErr(clEnqueueWriteBuffer(GetContext().queue, d_A, CL_TRUE, 0, (size_t)widthA * (size_t)heightA * sizeof(T), h_A, 0, NULL, NULL));
    clChkErr(clEnqueueWriteBuffer(GetContext().queue, d_B, CL_TRUE, 0, (size_t)widthB * (size_t)heightB * sizeof(T), h_B, 0, NULL, NULL));
}

template <class T>
void GemmOpenCL<T>::Launch()
{
    cl_int err;

    cl_uint cl_width0 = (cl_uint)widthA;
    cl_uint cl_width1 = (cl_uint)widthB;

    clChkErr(clSetKernelArg(GetContext().kernel, 0, sizeof(cl_mem), &d_A));
    clChkErr(clSetKernelArg(GetContext().kernel, 1, sizeof(cl_mem), &d_B));
    clChkErr(clSetKernelArg(GetContext().kernel, 2, sizeof(cl_mem), &d_C));
    clChkErr(clSetKernelArg(GetContext().kernel, 3, sizeof(cl_width0), &cl_width0));
    clChkErr(clSetKernelArg(GetContext().kernel, 4, sizeof(cl_width1), &cl_width1));

    size_t globalSize[3] = { 128, 128, 0 }; // grid size
    size_t localSize[3] = { 8, 8, 0 }; // block size

    err = clEnqueueNDRangeKernel(
        GetContext().queue,
        GetContext().kernel,
        2, NULL, globalSize, localSize, 0, NULL, NULL);

    clChkErr(clFinish(GetContext().queue));
}

template <class T>
void GemmOpenCL<T>::CopyFromDevice()
{
    clChkErr(clEnqueueReadBuffer(GetContext().queue, d_C, CL_TRUE, 0, (size_t)widthB * (size_t)heightA * sizeof(T), h_C, 0, NULL, NULL));
}

template <class T>
void GemmOpenCL<T>::FreeDevice()
{
    clChkErr(clReleaseMemObject(d_A));
    clChkErr(clReleaseMemObject(d_B));
    clChkErr(clReleaseMemObject(d_C));
}

