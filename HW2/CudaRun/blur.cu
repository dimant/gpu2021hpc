__global void blurKernel3x3(unsigned char* in, unsigned char* out, int w, int h)
{
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	if (Col < w && Row < h)
	{
		int pixVal = 0;
		int pixels = 0;

		for (int blurRow = -3; blurRow < -3 + 1; blurRow++)
		{
			for (int blurCol = -3; blurCol < -3 + 1; blurCol++)
			{
				int curRow = Row + blurRow;
				int curCol = Col + blurCol;

				if (curRow > -1 && curRow < h && curCol > -1 && curCol < w)
				{
					pixVal += in[curRow * w + curCol];
					pixels++;
				}
			}
		}

		out[Row * w + Col] = (unsigned char)(pixVal / pixels);
	}
}

__global void blurKernel9x9(unsigned char* in, unsigned char* out, int w, int h)
{
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	if (Col < w && Row < h)
	{
		int pixVal = 0;
		int pixels = 0;

		for (int blurRow = -9; blurRow < -9 + 1; blurRow++)
		{
			for (int blurCol = -9; blurCol < -9 + 1; blurCol++)
			{
				int curRow = Row + blurRow;
				int curCol = Col + blurCol;

				if (curRow > -1 && curRow < h && curCol > -1 && curCol < w)
				{
					pixVal += in[curRow * w + curCol];
					pixels++;
				}
			}
		}

		out[Row * w + Col] = (unsigned char)(pixVal / pixels);
	}
}

__global void blurKernelMxM(unsigned char* in, unsigned char* out, int w, int h, int M)
{
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int Row = blockIdx.y * blockDim.y + threadIdx.y;

	if (Col < w && Row < h)
	{
		int pixVal = 0;
		int pixels = 0;

		for (int blurRow = -M; blurRow < -M + 1; blurRow++)
		{
			for (int blurCol = -M; blurCol < -M + 1; blurCol++)
			{
				int curRow = Row + blurRow;
				int curCol = Col + blurCol;

				if (curRow > -1 && curRow < h && curCol > -1 && curCol < w)
				{
					pixVal += in[curRow * w + curCol];
					pixels++;
				}
			}
		}

		out[Row * w + Col] = (unsigned char)(pixVal / pixels);
	}
}
