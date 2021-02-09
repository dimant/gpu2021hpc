#include <iostream>

#include "ocl_util.h"
#include "BlurOpenCL.h"

void BlurOpenCL::AllocateDevice()
{
    cl_int err;
    size_t size = rows * cols * sizeof(char);

    d_A = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, size, NULL, &err); clChkErr(err);
    d_B = clCreateBuffer(GetContext().context, CL_MEM_READ_ONLY, size, NULL, &err); clChkErr(err);
}

void BlurOpenCL::CopyToDevice()
{
    size_t size = rows * cols * sizeof(char);

    clChkErr(clEnqueueWriteBuffer(GetContext().queue, d_A, CL_TRUE, 0, size, h_A, 0, NULL, NULL));
    clChkErr(clEnqueueWriteBuffer(GetContext().queue, d_B, CL_TRUE, 0, size, h_B, 0, NULL, NULL));
}

void BlurOpenCL::Launch()
{
    cl_int err;

    cl_int cl_nrows = (cl_int)rows;
    cl_int cl_ncols = (cl_int)cols;
    cl_int cl_blurSize = (cl_int)blurSize;

    clChkErr(clSetKernelArg(GetContext().kernel, 0, sizeof(cl_mem), &d_A));
    clChkErr(clSetKernelArg(GetContext().kernel, 1, sizeof(cl_mem), &d_B));
    clChkErr(clSetKernelArg(GetContext().kernel, 2, sizeof(cl_nrows), &cl_nrows));
    clChkErr(clSetKernelArg(GetContext().kernel, 3, sizeof(cl_ncols), &cl_ncols));

    if (blurSize != 1 && blurSize != 4)
    {
        clChkErr(clSetKernelArg(GetContext().kernel, 4, sizeof(cl_blurSize), &cl_blurSize));
    }

    size_t globalSize[3] = { 128, 128, 0 }; // grid size
    size_t localSize[3] = { 8, 8, 0 }; // block size

    err = clEnqueueNDRangeKernel(
        GetContext().queue,
        GetContext().kernel,
        2, NULL, globalSize, localSize, 0, NULL, NULL);

    clChkErr(clFinish(GetContext().queue));
}

void BlurOpenCL::CopyFromDevice()
{
    size_t size = rows * cols * sizeof(char);

    clChkErr(clEnqueueReadBuffer(GetContext().queue, d_B, CL_TRUE, 0, size, h_B, 0, NULL, NULL));
}

void BlurOpenCL::FreeDevice()
{
    clChkErr(clReleaseMemObject(d_A));
    clChkErr(clReleaseMemObject(d_B));
}