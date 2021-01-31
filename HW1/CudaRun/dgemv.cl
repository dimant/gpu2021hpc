#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void dgemv(const double alpha, __global double* A, __global double* x, const double beta, __global double* y, __global double* z, const uint rows, const uint cols)
{
    int stride = get_global_size(0);
    double buf;

    for (int row = get_global_id(0); row < rows; row += stride)
    {
        buf = 0.0;

        for (int col = 0; col < cols; col++)
        {
            buf += alpha * A[row * cols + col] * x[col];
        }

        z[row] = buf + beta * y[row];
    }
}

