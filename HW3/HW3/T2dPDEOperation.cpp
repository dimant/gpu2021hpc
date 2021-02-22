#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "T2dPDEOperation.h"

#include "finite_difference_ref.h"

void T2dPDEOperation::AllocateHost()
{
	h_temp_in = new float[nrows * ncols];
	t_temp_in = new float[nrows * ncols];
	h_temp_out = new float[nrows * ncols];
	t_temp_out = new float[nrows * ncols];
}

inline float random_temp()
{
	float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	float t = (r * 40) + 80; // random temperature between 80C and 120C
	//float t = (float)rand() / (float)(RAND_MAX / 100.0f);
	return t;
}

void T2dPDEOperation::InitData()
{
	srand(time(NULL));

	for (int i = 0; i < nrows * ncols; i++)
	{
		float t = random_temp();
		h_temp_in[i] = t;
		t_temp_in[i] = t;
		h_temp_out[i] = t;
		t_temp_out[i] = t;
	}
}

void T2dPDEOperation::VerifyResult()
{
	if (std::string(reference_impl) == "center")
	{
		center_diff((int)steps, (int)ncols, (int)nrows, alpha, t_temp_in, t_temp_out);
	}
	else if (std::string(reference_impl) == "center_clamp")
	{
		center_clamp_diff((int)steps, (int)ncols, (int)nrows, alpha, t_temp_in, t_temp_out);
	}
	else if (std::string(reference_impl) == "full")
	{
		full_diff((int)steps, (int)ncols, (int)nrows, alpha, t_temp_in, t_temp_out);
	}
	else
	{
		std::cerr << "Unknown reference implementation for PDE: " << reference_impl << std::endl;
	}

	float maxError = FLT_MIN;
	float* t_temp = nullptr;

	if (steps % 2 == 0)
	{
		t_temp = t_temp_in;
	}
	else
	{
		t_temp = t_temp_out;
	}

	for (int row = 0; row < nrows; row++)
	{
		for (int col = 0; col < ncols; col++)
		{
			int idx = row * ncols + col;
			float delta = fabs(h_temp_out[idx] - t_temp[idx]);

			if (delta > maxError)
			{
				maxError = delta;
			}

		}
	}

	// Check and see if our maxError is greater than an error bound
	if (maxError > 1.0e-30f)
	{
		printf("Problem! The Max Error of %.5e is NOT within acceptable bounds.\n", maxError);
	}
	else
	{
		printf("The Max Error of %.5e is within acceptable bounds.\n", maxError);
	}
}

void T2dPDEOperation::FreeHost()
{
	delete[] h_temp_in;
	delete[] t_temp_in;
	delete[] h_temp_out;
	delete[] t_temp_out;
}
