#ifndef MATADD_OPERATION_H
#define MATADD_OPERATION_H

#include <cuda.h>

#include "HpcOperation.h"

class MatAddOperation : public HpcOperation<CUfunction>
{
private:
	size_t nrows;
	size_t ncols;

	float* h_A;
	float* h_B;
	float* h_C;

	CUdeviceptr d_A;
	CUdeviceptr d_B;
	CUdeviceptr d_C;

public:
	MatAddOperation(size_t nr, size_t nc) :
		nrows(nr), ncols(nc),
		h_A(nullptr), h_B(nullptr), h_C(nullptr),
		d_A(0), d_B(0), d_C(0)
	{
	}

	size_t GetSize()
	{
		return nrows * ncols * sizeof(float);
	}

	void AllocateHost();
	void AllocateDevice();
	void InitData();
	void CopyToDevice();
	void Launch();
	void CopyFromDevice();
	void VerifyResult();
	void FreeHost();
	void FreeDevice();
};

#endif