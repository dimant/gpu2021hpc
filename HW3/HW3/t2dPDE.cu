// index into (flattened) 1D array from 2D space
// num = number of columns per row
#define I2D(ncols, col, row) ((row)*(ncols)+(col))

__global__
void t2dPDE_forward_step(int ncols, int nrows, float alpha, float* temp_in, float* temp_out)
{
    int i00, ip20, ip10, i0p2, i0p1;
    float d2tdx2, d2tdy2;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // loop over all points in domain (except boundary)
    if (col > 0 && col > 0 && col < ncols - 1 && row < nrows - 1)
    {
        // find indices into linear memory
        // for central point and neighbours
        i00 =  I2D(ncols, col, row);
        ip20 = I2D(ncols, col + 2, row);
        ip10 = I2D(ncols, col + 1, row);
        i0p2 = I2D(ncols, col, row + 2);
        i0p1 = I2D(ncols, col, row + 1);

        // evaluate derivatives
        d2tdx2 = temp_in[ip20] - 2 * temp_in[ip10] + temp_in[i00];
        d2tdy2 = temp_in[i0p2] - 2 * temp_in[i0p1] + temp_in[i00];

        // update temperatures
        temp_out[i00] = temp_in[i00] + alpha * (d2tdx2 + d2tdy2);
    }
}

void t2PDE_forward(int nsteps, int ncols, int nrows, float alpha, float* temp_in, float* temp_out)
{
    float* left = temp_in;
    float* right = temp_out;
    float* swap;

    for (int i = 0; i < nsteps; i++)
    {
        t2PDE_forward_step(ncols, nrows, alpha, left, right);
        __syncthreads();

        swap = left;
        left = right;
        right = left;
    }
}

__global__
void t2dPDE_center_step(int ncols, int nrows, float alpha, float* temp_in, float* temp_out)
{
    int i00, im10, ip10, i0m1, i0p1;
    float d2tdx2, d2tdy2;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // loop over all points in domain (except boundary)
    if (col > 0 && col > 0 && col < ncols - 1 && row < nrows - 1)
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

void t2PDE_center(int nsteps, int ncols, int nrows, float alpha, float* temp_in, float* temp_out)
{
    float* left = temp_in;
    float* right = temp_out;
    float* swap;

    for (int i = 0; i < nsteps; i++)
    {
        t2PDE_center_step(ncols, nrows, alpha, left, right);
        __syncthreads();

        swap = left;
        left = right;
        right = left;
    }
}