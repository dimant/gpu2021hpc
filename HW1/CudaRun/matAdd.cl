#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_intel_printf : enable

__kernel void matAdd(__global float* A, __global float* B, __global float* C, const ulong nrows, const ulong ncols)
{
    int stridex = get_local_size(0) * get_num_groups(0);
    int stridey = get_local_size(1) * get_num_groups(1);

    for (int row = get_global_id(1); row < nrows; row += stridey)
    {
        for (int col = get_global_id(0); col < ncols; col += stridex)
        {
            int idx = row * ncols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}
