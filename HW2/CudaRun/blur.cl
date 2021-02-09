__kernel void blur3x3(__global unsigned char* A, __global unsigned char* B, const int rows, const int cols)
{
	int col = get_global_id(0);
	int row = get_global_id(1);

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

__kernel void blur9x9(__global unsigned char* A, __global unsigned char* B, const int rows, const int cols)
{
	int col = get_global_id(0);
	int row = get_global_id(1);

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