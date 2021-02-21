#include "finite_difference_ref.h"


#define I2D(ncols, col, row) ((row)*(ncols)+(col))

inline float d2tdx2_center(int ncols, int col, int row, float* temp_in)
{
	int i00, im10, ip10;

	i00 = I2D(ncols, col, row);
	im10 = I2D(ncols, col - 1, row);
	ip10 = I2D(ncols, col + 1, row);

	return temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
}

inline float d2tdy2_center(int ncols, int col, int row, float* temp_in)
{
	int i00, i0m1, i0p1;

	i00 = I2D(ncols, col, row);
	i0m1 = I2D(ncols, col, row - 1);
	i0p1 = I2D(ncols, col, row + 1);

	return temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];
}

inline float d2tdx2_forward(int ncols, int col, int row, float* temp_in)
{
	int ip20, ip10, i00;

	i00 = I2D(ncols, col, row);
	ip20 = I2D(ncols, col + 2, row);
	ip10 = I2D(ncols, col + 1, row);

	return temp_in[ip20] - 2 * temp_in[ip10] + temp_in[i00];
}

inline float d2tdy2_forward(int ncols, int col, int row, float* temp_in)
{
	int i00, i0p2, i0p1;

	i00 = I2D(ncols, col, row);
	i0p2 = I2D(ncols, col, row + 2);
	i0p1 = I2D(ncols, col, row + 1);

	return temp_in[i0p2] - 2 * temp_in[i0p1] + temp_in[i00];
}

inline float d2tdx2_backward(int ncols, int col, int row, float* temp_in)
{
	int i00, im10, im20;

	i00 = I2D(ncols, col, row);
	im10 = I2D(ncols, col - 1, row);
	im20 = I2D(ncols, col - 2, row);

	return temp_in[i00] - 2 * temp_in[im10] + temp_in[im20];
}

inline float d2tdy2_backward(int ncols, int col, int row, float* temp_in)
{
	int i00, i0m1, i0m2;

	i00 = I2D(ncols, col, row);
	i0m1 = I2D(ncols, col, row - 1);
	i0m2 = I2D(ncols, col, row - 2);

	return temp_in[i00] - 2 * temp_in[i0m1] + temp_in[i0m2];
}

void step_center(int nrows, int ncols, float alpha, float* temp_in, float* temp_out)
{
	int i00;

	for (int row = 1; row < nrows - 1; row++) {
		for (int col = 1; col < ncols - 1; col++) {
			i00 = I2D(ncols, col, row);

			temp_out[i00] = temp_in[i00] + alpha *
				(d2tdx2_center(nrows, col, row, temp_in) +
				 d2tdy2_center(nrows, col, row, temp_in));
		}
	}
}

void step_forward(int nrows, int ncols, float alpha, float* temp_in, float* temp_out)
{
	int i00;

	for (int row = 0; row < nrows - 2; row++)
	{
		for (int col = 0; col < ncols - 2; col++)
		{
			i00  = I2D(ncols, col, row);

			temp_out[i00] = temp_in[i00] + alpha *
				(d2tdx2_forward(nrows, col, row, temp_in) +
				 d2tdy2_forward(nrows, col, row, temp_in));
		}
	}
}

void step_backward(int nrows, int ncols, float alpha, float* temp_in, float* temp_out)
{
	int i00;

	for (int row = 2; row < nrows; row++)
	{
		for (int col = 2; col < ncols; col++)
		{
			i00  = I2D(ncols, col, row);

			temp_out[i00] = temp_in[I2D(ncols, col, row)] + alpha *
				(d2tdx2_backward(nrows, col, row, temp_in) +
				 d2tdy2_backward(nrows, col, row, temp_in));
		}
	}
}