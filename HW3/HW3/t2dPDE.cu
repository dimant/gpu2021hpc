// index into (flattened) 1D array from 2D space
// num = number of columns per row
#define I2D(ncols, col, row) ((row)*(ncols)+(col))

extern "C" __global__
void t2dPDE_center_step(int ncols, int nrows, float alpha, float* temp_in, float* temp_out)
{
    int i00, im10, ip10, i0m1, i0p1;
    float d2tdx2, d2tdy2;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // loop over all points in domain (except boundary)
    if (col > 0 && col < ncols - 1 && 
        row > 0 && row < nrows - 1)
    {
        // find indices into linear memory
        // for central point and neighbours
        i00 = I2D(ncols, col, row);
        ip10 = I2D(ncols, col + 1, row);
        im10 = I2D(ncols, col - 1, row);
        i0p1 = I2D(ncols, col, row + 1);
        i0m1 = I2D(ncols, col, row - 1);

        // evaluate derivatives
        d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
        d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];

        // update temperatures
        temp_out[i00] = temp_in[i00] + alpha * (d2tdx2 + d2tdy2);
    }
}

extern "C" __global__
void t2dPDE_center(int threads, int blocks,
    int nsteps, int ncols, int nrows, float alpha, float* temp_in, float* temp_out)
{
    dim3 gridSize(blocks, blocks);
    dim3 blockSize(threads, threads);

    float* left = temp_in;
    float* right = temp_out;
    float* swap = 0;

    for (int i = 0; i < nsteps; i++)
    {
        t2dPDE_center_step<<<gridSize, blockSize>>>(ncols, nrows, alpha, left, right);
        cudaDeviceSynchronize();

        swap = left;
        left = right;
        right = swap;
    }
}