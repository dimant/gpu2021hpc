#pragma once

class T2dPDEOperation
{
protected:
	float* h_temp_in;
	float* h_temp_out;

	// used for validation
	float* t_temp_in;
	float* t_temp_out;

	float alpha;
	size_t nrows;
	size_t ncols;
	size_t steps;

	const char* reference_impl;

public:
	T2dPDEOperation(size_t steps, size_t nrows, size_t ncols,
		float alpha, const char* reference_impl) :
		steps(steps), nrows(nrows), ncols(ncols), alpha(alpha),
		h_temp_in(nullptr), h_temp_out(nullptr),
		t_temp_in(nullptr), t_temp_out(nullptr),
		reference_impl(reference_impl)
	{}

	void AllocateHost();
	void InitData();
	void VerifyResult();
	void FreeHost();
};