#ifndef DGEMVCUDA_H
#define DGEMVCUDA_H

#include <cuda.h>

#include "HpcOperation.h"
#include "DgemvOperation.h"

class DgemvCuda: public HpcOperation<CUfunction>, public DgemvOperation
{
private:
	CUdeviceptr d_A;
	CUdeviceptr d_x;
	CUdeviceptr d_y;
	CUdeviceptr d_z;

public:
	DgemvCuda(size_t rows, size_t cols) :
		d_A(0), d_x(0), d_y(0), d_z(0), DgemvOperation(rows, cols)
	{
	}

	void AllocateHost()
	{
		DgemvOperation::AllocateHost();
	}

	void InitData()
	{
		DgemvOperation::InitData();
	}

	void VerifyResult()
	{
		DgemvOperation::VerifyResult();
	}

	void FreeHost()
	{
		DgemvOperation::FreeHost();
	}

	void Launch();
	void CopyToDevice();
	void CopyFromDevice();
	void AllocateDevice();
	void FreeDevice();
};

#endif