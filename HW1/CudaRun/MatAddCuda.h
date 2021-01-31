#ifndef MATADDCUDA_H
#define MATADDCUDA_H

#include <cuda.h>

#include "HpcOperation.h"
#include "MatAddOperation.h"

class MatAddCuda : public HpcOperation<CUfunction>, public MatAddOperation
{
private:
	CUdeviceptr d_A;
	CUdeviceptr d_B;
	CUdeviceptr d_C;

public:
	MatAddCuda(size_t nr, size_t nc) :
		d_A(0), d_B(0), d_C(0), MatAddOperation(nr, nc)
	{
	}

	void AllocateHost()
	{
		MatAddOperation::AllocateHost();
	}

	void InitData()
	{
		MatAddOperation::InitData();
	}

	void VerifyResult()
	{
		MatAddOperation::VerifyResult();
	}

	void FreeHost()
	{
		MatAddOperation::FreeHost();
	}

	void Launch();
	void CopyFromDevice();
	void CopyToDevice();
	void AllocateDevice();
	void FreeDevice();
};

#endif