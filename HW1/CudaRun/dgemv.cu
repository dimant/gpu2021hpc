extern "C" __global__ void dgemv(
    const double alpha,
    const double* A,
    const double* x,
    const double beta,
    const double* y,
    double* z,
    size_t rows, size_t cols)
{
    int stride = blockDim.x * gridDim.x;
    double buf;

    for (int row = blockDim.x * blockIdx.x + threadIdx.x; row < rows; row += stride)
    {
        buf = 0.0;

        for (int col = 0; col < cols; col++)
        {
            buf += alpha * A[row * cols + col] * x[col];
        }

        z[row] = buf + beta * y[row];
    }
}