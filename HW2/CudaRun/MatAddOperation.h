#ifndef MATADD_OPERATION_H
#define MATADD_OPERATION_H

class MatAddOperation
{
protected:
	size_t nrows;
	size_t ncols;

	float* h_A;
	float* h_B;
	float* h_C;

public:
	MatAddOperation(size_t nr, size_t nc) :
		nrows(nr), ncols(nc),
		h_A(nullptr), h_B(nullptr), h_C(nullptr)
	{
	}

	size_t GetSize()
	{
		return nrows * ncols * sizeof(float);
	}

	void AllocateHost();
	void InitData();
	void VerifyResult();
	void FreeHost();
};

#endif