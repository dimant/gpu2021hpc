#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void matAdd(__global float* A, __global float* B, __global float* C, const uint nrows, const uint ncols)
{
    int stridex = get_global_size(0);
    int stridey = get_global_size(1);

    for (int row = get_global_id(1); row < nrows; row += stridey)
    {
        for (int col = get_global_id(0); col < ncols; col += stridex)
        {
            int idx = row * ncols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

__kernel void matAddRow(__global float* A, __global float* B, __global float* C, const uint nrows, const uint ncols)
{
    int stridey = get_global_size(1);

    for (int row = get_global_id(1); row < nrows; row += stridey)
    {
        for (int col = 0; col < ncols; col++)
        {
            int idx = row * ncols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

__kernel void matAddCol(__global float* A, __global float* B, __global float* C, const uint nrows, const uint ncols)
{
    int stridex = get_global_size(0);

    for (int col = get_global_id(0); col < ncols; col += stridex)
    {
        for (int row = 0; row < nrows; row++)
        {
            int idx = row * ncols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}
