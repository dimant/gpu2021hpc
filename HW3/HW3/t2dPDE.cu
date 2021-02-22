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

    if (col < ncols && row < nrows)
    {
        i00 = I2D(ncols, col, row);

        if (col == 0)
        {
            ip10 = I2D(ncols, col + 1, row);
            d2tdx2 = 0.0f - 2 * temp_in[i00] + temp_in[ip10];
        }
        else if (col == ncols - 1)
        {
            im10 = I2D(ncols, col - 1, row);
            d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + 0.0f;
        }
        else
        {
            ip10 = I2D(ncols, col + 1, row);
            im10 = I2D(ncols, col - 1, row);
            d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
        }

        if (row == 0)
        {
            i0p1 = I2D(ncols, col, row + 1);
            d2tdy2 = 0.0f - 2 * temp_in[i00] + temp_in[i0p1];
        }
        else if (row == nrows - 1)
        {
            i0m1 = I2D(ncols, col, row - 1);
            d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + 0.0f;
        }
        else
        {
            i0p1 = I2D(ncols, col, row + 1);
            i0m1 = I2D(ncols, col, row - 1);
            d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];
        }

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

extern "C" __global__
void t2dPDE_full_step(int ncols, int nrows, float alpha, float* temp_in, float* temp_out)
{
    int i00, im10, ip10, i0m1, i0p1, ip20, im20, i0p2, i0m2;
    float d2tdx2, d2tdy2;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < ncols && row < nrows)
    {
        i00 = I2D(ncols, col, row);

        if (col == 0)
        {
            ip10 = I2D(ncols, col + 1, row);
            ip20 = I2D(ncols, col + 2, row);
            d2tdx2 = temp_in[ip20] - 2 * temp_in[ip10] + temp_in[i00];
        }
        else if (col == ncols - 1)
        {
            im10 = I2D(ncols, col - 1, row);
            im20 = I2D(ncols, col - 2, row);
            d2tdx2 = temp_in[i00] - 2 * temp_in[im10] + temp_in[im20];
        }
        else
        {
            ip10 = I2D(ncols, col + 1, row);
            im10 = I2D(ncols, col - 1, row);
            d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
        }

        if (row == 0)
        {
            i0p1 = I2D(ncols, col, row + 1);
            i0p2 = I2D(ncols, col, row + 2);
            d2tdy2 = temp_in[i0p2] - 2 * temp_in[i0p1] + temp_in[i00];
        }
        else if (row == nrows - 1)
        {
            i0m1 = I2D(ncols, col, row - 1);
            i0m2 = I2D(ncols, col, row - 2);
            d2tdy2 = temp_in[i00] - 2 * temp_in[i0m1] + temp_in[i0m2];
        }
        else
        {
            i0p1 = I2D(ncols, col, row + 1);
            i0m1 = I2D(ncols, col, row - 1);
            d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];
        }

        // update temperatures
        temp_out[i00] = temp_in[i00] + alpha * (d2tdx2 + d2tdy2);
    }
}

extern "C" __global__
void t2dPDE_full(int threads, int blocks,
    int nsteps, int ncols, int nrows, float alpha, float* temp_in, float* temp_out)
{
    dim3 gridSize(blocks, blocks);
    dim3 blockSize(threads, threads);

    float* left = temp_in;
    float* right = temp_out;
    float* swap = 0;

    for (int i = 0; i < nsteps; i++)
    {
        t2dPDE_full_step << <gridSize, blockSize >> > (ncols, nrows, alpha, left, right);
        cudaDeviceSynchronize();

        swap = left;
        left = right;
        right = swap;
    }
}