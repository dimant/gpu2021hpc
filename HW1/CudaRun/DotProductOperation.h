#ifndef DOTPRODUCTOPERATION_H
#define DOTPRODUCTOPERATION_H

#include "CudaOperation.h"

template <class T>
class DotProductOperation : public CudaOperation
{
private:
	const int threadsPerBlock = 256;
	const int blocksPerGrid = 16;

	size_t elements;

	T* h_A;
	T* h_B;
	float* h_C;

	CUdeviceptr d_A;
	CUdeviceptr d_B;
	CUdeviceptr d_C;

public:
	DotProductOperation(size_t e) :
		elements(e),
		h_A(nullptr), h_B(nullptr), h_C(nullptr),
		d_A(0), d_B(0), d_C(0)
	{
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