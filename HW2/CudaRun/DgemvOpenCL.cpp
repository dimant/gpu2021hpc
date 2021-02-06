#include "DgemvOpenCL.h"

#include <iostream>

#include "ocl_util.h"

void DgemvOpenCL::AllocateDevice()
{
    cl_int err;

    d_A = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, rows * cols * sizeof(double), NULL, &err); clChkErr(err);
    d_x = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, cols * sizeof(double), NULL, &err); clChkErr(err);
    d_y = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, rows * sizeof(double), NULL, &err); clChkErr(err);
    d_z = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, rows * sizeof(double), NULL, &err); clChkErr(err);

    unsigned char zero = '\0';
    clChkErr(clEnqueueFillBuffer(GetContext().queue, d_z, &zero, 1, 0, rows * sizeof(double), NULL, 0, NULL));
}

void DgemvOpenCL::CopyToDevice()
{
    clChkErr(clEnqueueWriteBuffer(GetContext().queue, d_A, CL_TRUE, 0, rows * cols * sizeof(double), h_A, 0, NULL, NULL));
    clChkErr(clEnqueueWriteBuffer(GetContext().queue, d_x, CL_TRUE, 0, cols * sizeof(double), h_A, 0, NULL, NULL));
    clChkErr(clEnqueueWriteBuffer(GetContext().queue, d_y, CL_TRUE, 0, rows * sizeof(double), h_A, 0, NULL, NULL));
}

void DgemvOpenCL::Launch()
{
    cl_int err;

    cl_uint cl_rows = (cl_uint)rows;
    cl_uint cl_cols = (cl_uint)cols;

    clChkErr(clSetKernelArg(GetContext().kernel, 0, sizeof(alpha), &alpha));
    clChkErr(clSetKernelArg(GetContext().kernel, 1, sizeof(cl_mem), &d_A));
    clChkErr(clSetKernelArg(GetContext().kernel, 2, sizeof(cl_mem), &d_x));
    clChkErr(clSetKernelArg(GetContext().kernel, 3, sizeof(beta), &beta));
    clChkErr(clSetKernelArg(GetContext().kernel, 4, sizeof(cl_mem), &d_y));
    clChkErr(clSetKernelArg(GetContext().kernel, 5, sizeof(cl_mem), &d_z));
    clChkErr(clSetKernelArg(GetContext().kernel, 6, sizeof(cl_rows), &cl_rows));
    clChkErr(clSetKernelArg(GetContext().kernel, 7, sizeof(cl_cols), &cl_cols));

    size_t globalSize[3] = { 128, 128, 0 }; // grid size
    size_t localSize[3] = { 8, 8, 0 }; // block size

    err = clEnqueueNDRangeKernel(
        GetContext().queue,
        GetContext().kernel,
        2, NULL, globalSize, localSize, 0, NULL, NULL);

    clChkErr(clFinish(GetContext().queue));
}

void DgemvOpenCL::CopyFromDevice()
{
    clChkErr(clEnqueueReadBuffer(GetContext().queue, d_z, CL_TRUE, 0, rows * sizeof(double), h_z, 0, NULL, NULL));
}

void DgemvOpenCL::FreeDevice()
{
    clChkErr(clReleaseMemObject(d_A));
    clChkErr(clReleaseMemObject(d_x));
    clChkErr(clReleaseMemObject(d_y));
    clChkErr(clReleaseMemObject(d_z));
}

