extern "C" __global__ void blur3x3(unsigned char* A, unsigned char* B, int rows, int cols)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows)
	{
		int pixVal = 0;
		int pixels = 0;

		for (int blurRow = -1; blurRow < 1 + 1; blurRow++)
		{
			for (int blurCol = -1; blurCol < 1 + 1; blurCol++)
			{
				int curRow = row + blurRow;
				int curCol = col + blurCol;

				if (curRow > -1 && curRow < rows && curCol > -1 && curCol < cols)
				{
					pixVal += A[curRow * cols + curCol];
					pixels++;
				}
			}
		}

		B[row * cols + col] = (unsigned char)(pixVal / pixels);
	}
}

extern "C" __global__ void blur9x9(unsigned char* A, unsigned char* B, int rows, int cols)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col < cols && row < rows)
	{
		int pixVal = 0;
		int pixels = 0;

		for (int blurRow = -4; blurRow < 4 + 1; blurRow++)
		{
			for (int blurCol = -4; blurCol < 4 + 1; blurCol++)
			{
				int curRow = row + blurRow;
				int curCol = col + blurCol;

				if (curRow > -1 && curRow < rows && curCol > -1 && curCol < cols)
				{
					pixVal += A[curRow * cols + curCol];
					pixels++;
				}
			}
		}

		B[row * cols + col] = (unsigned char)(pixVal / pixels);
	}
}
