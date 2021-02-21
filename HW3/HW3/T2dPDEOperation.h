#pragma once

class T2dPDEOperation
{
protected:
	float* h_temp_in;
	float* h_temp_out;
	float alpha;
	size_t nrows;
	size_t ncols;
	size_t steps;

public:
	T2dPDEOperation(size_t steps, size_t nrows, size_t ncols,
		float alpha) :
		steps(steps), nrows(nrows), ncols(ncols), alpha(alpha),
		h_temp_in(nullptr), h_temp_out(nullptr)
	{}

	void AllocateHost();
	void InitData();
	void VerifyResult();
	void FreeHost();
};