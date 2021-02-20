extern "C" __global__ void imageFilterSingle(
    float* d_output, float* d_input, float* d_filter,
    int num_row, int num_col, int filter_size)
{
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx_x < num_col && idx_y < num_row)
    {
        float result = 0.f;
        for (int filter_row = -filter_size / 2; filter_row <= filter_size / 2; ++filter_row)
        {
            for (int filter_col = -filter_size / 2; filter_col <= filter_size / 2; ++filter_col)
            {
                int image_row = idx_y + filter_row;
                int image_col = idx_x + filter_col;
                float filter_value = d_filter[(filter_row + filter_size / 2) * filter_size + filter_col + filter_size / 2];
                float image_value;

                if (image_row >= 0 && image_row < num_row && image_col >= 0 && image_col < num_col)
                {
                    image_value = d_input[image_row * num_col + image_col];
                }
                else
                {
                    image_value = 0.f;
                }

                result += image_value * filter_value;
            }

        }

        d_output[idx_y * num_col + idx_x] = result;
    }
}

extern "C" __global__ void imageFilter(
    float* d_output, float* d_input,
    int num_row, int num_col,
    float* filters, int* filter_sizes, int nfilters)
{
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;

    if (idx_x < num_col && idx_y < num_row)
    {
        float* cursor = filters;

        for (int i = 0; i < nfilters; i++)
        {
            int filterSize = filter_sizes[i];

            imageFilterSingle(d_output, d_input, cursor, num_row, num_col, filterSize);

            cursor += filterSize * filterSize;

            __syncthreads();
        }
    }
}
