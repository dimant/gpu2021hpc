__kernel void blurKernel3x3(unsigned char* in, unsigned char* out, int w, int h)
{
	int col = get_global_id(0);
	int row = get_global_id(1);

	if (col < w && row < h)
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

		out[row * w + col] = (unsigned char)(pixVal / pixels);
	}
}
