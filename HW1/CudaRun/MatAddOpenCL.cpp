#include <iostream>

#include "ocl_util.h"
#include "MatAddOpenCL.h"

void MatAddOpenCL::AllocateDevice()
{
    cl_int err;
    size_t size = GetSize();

    d_A = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, size, NULL, &err); clChkErr(err);
    d_B = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, size, NULL, &err); clChkErr(err);
    d_C = clCreateBuffer(GetContext().context, CL_MEM_WRITE_ONLY, size, NULL, &err); clChkErr(err);
}

void MatAddOpenCL::CopyToDevice()
{
    size_t size = GetSize();

    clChkErr(clEnqueueWriteBuffer(GetContext().queue, d_A, CL_TRUE, 0, size, h_A, 0, NULL, NULL));
    clChkErr(clEnqueueWriteBuffer(GetContext().queue, d_B, CL_TRUE, 0, size, h_B, 0, NULL, NULL));
}

void MatAddOpenCL::Launch()
{
    cl_int err;

    cl_uint cl_nrows = (cl_uint)nrows;
    cl_uint cl_ncols = (cl_uint)ncols;

    clChkErr(clSetKernelArg(GetContext().kernel, 0, sizeof(cl_mem), &d_A));
    clChkErr(clSetKernelArg(GetContext().kernel, 1, sizeof(cl_mem), &d_B));
    clChkErr(clSetKernelArg(GetContext().kernel, 2, sizeof(cl_mem), &d_C));
    clChkErr(clSetKernelArg(GetContext().kernel, 3, sizeof(cl_nrows), &cl_nrows));
    clChkErr(clSetKernelArg(GetContext().kernel, 4, sizeof(cl_ncols), &cl_ncols));

    size_t globalSize[3] = { 128, 128, 0 }; // grid size
    size_t localSize[3] = { 8, 8, 0 }; // block size

    err = clEnqueueNDRangeKernel(
        GetContext().queue,
        GetContext().kernel,
        2, NULL, globalSize, localSize, 0, NULL, NULL);

    clChkErr(clFinish(GetContext().queue));
}

void MatAddOpenCL::CopyFromDevice()
{
    size_t size = GetSize();

    clChkErr(clEnqueueReadBuffer(GetContext().queue, d_C, CL_TRUE, 0, size, h_C, 0, NULL, NULL));
}

void MatAddOpenCL::FreeDevice()
{
    clChkErr(clReleaseMemObject(d_A));
    clChkErr(clReleaseMemObject(d_B));
    clChkErr(clReleaseMemObject(d_C));
}