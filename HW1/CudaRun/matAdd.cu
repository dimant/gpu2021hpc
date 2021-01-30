// https://forums.developer.nvidia.com/t/why-am-i-getting-better-performance-with-per-column-vs-per-row-for-matrix-addition/48774
//
// This has to do with memory coalescing in CUDA, i.e. efficient use of the memory subsystem.
// When each thread is reading a column of data, then adjacent threads in a warp, at 
// each memory read instruction, are loading adjacent data from memory.This is the most 
// optimal usage of the memory subsystem.
// When each thread is reading a row of data, then adjacent threads in a warp are requesting 
// data that is separated by the row width.This is less efficient.
// This presentation may be of interest : 
// http://on-demand.gputechconf.com/gtc/2012/presentations/S0514-GTC2012-GPU-Performance-Analysis.pdf 25
// It’s necessary to think about what adjacent threads in a warp are doing instruction - by - 
// instruction, in order to understand coalescing.

extern "C" __global__ void matAdd(const float* A, const float* B, float* C, int nrows, int ncols)
{
    int stridex = blockDim.x * gridDim.x;
    int stridey = blockDim.y * gridDim.y;

    for (int row = blockDim.y * blockIdx.y + threadIdx.y; row < nrows; row += stridey)
    {
        for (int col = blockDim.x * blockIdx.x + threadIdx.x; col < ncols; col += stridex)
        {
            int idx = row * ncols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

extern "C" __global__ void matAddRow(const float* A, const float* B, float* C, int nrows, int ncols)
{
    int stridey = blockDim.y * gridDim.y;

    for (int row = blockDim.y * blockIdx.y + threadIdx.y; row < nrows; row += stridey)
    {
        for (int col = 0; col < ncols; col++)
        {
            int idx = row * ncols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

extern "C" __global__ void matAddCol(const float* A, const float* B, float* C, int nrows, int ncols)
{
    int stridex = blockDim.x * gridDim.x;

    for (int col = blockDim.x * blockIdx.x + threadIdx.x; col < ncols; col += stridex)
    {
        for (int row = 0; row < nrows; row++)
        {
            int idx = row * ncols + col;
            C[idx] = A[idx] + B[idx];
        }
    }
}

