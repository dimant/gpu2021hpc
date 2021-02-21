#include <iostream>
#include <stdlib.h>
#include <math.h>

#include "T2dPDEOperation.h"

#include "finite_difference_ref.h"

void T2dPDEOperation::AllocateHost()
{
	h_temp_in = new float[nrows * ncols];
	t_temp_in = new float[nrows * ncols];
	h_temp_out = new float[nrows * ncols];
	t_temp_out = new float[nrows * ncols];
}

float random_temp()
{
	float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	float t = (r * 40) + 80; // random temperature between 80C and 120C
	return t;
}

void T2dPDEOperation::InitData()
{
	for (int i = 0; i < nrows * ncols; i++)
	{
		h_temp_in[i] = random_temp();
		t_temp_in[i] = h_temp_in[i];
		h_temp_out[i] = random_temp();
		t_temp_out[i] = h_temp_out[i];
	}
}

void T2dPDEOperation::VerifyResult()
{
	center_diff((int)steps, (int)ncols, (int)nrows, alpha, t_temp_in, t_temp_out);

	// Output should always be stored in the h_temp_out and t_temp_in at this point
	// we ensure h_temp_out has the output by copying correctly from the device
	// we ensure t_temp_in has the output by passing the pointers by reference
	float maxError = FLT_MIN;

	for (int i = 0; i < nrows * ncols; ++i) {
		float delta = fabs(h_temp_out[i] - t_temp_in[i]);

		if (delta > maxError)
		{
			maxError = delta;
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
