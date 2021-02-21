#include <stdlib.h>

#include "T2dPDEOperation.h"


void T2dPDEOperation::AllocateHost()
{
	h_temp_in = new float[nrows * ncols];
	h_temp_out = new float[nrows * ncols];
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
		h_temp_out[i] = random_temp();
	}
}

void T2dPDEOperation::VerifyResult()
{

}

void T2dPDEOperation::FreeHost()
{
	delete[] h_temp_in;
	delete[] h_temp_out;
}
