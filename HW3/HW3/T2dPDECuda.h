#pragma once

#include "CudaModule.h"
#include "HpcOperation.h"
#include "T2dPDEOperation.h"

class T2dPDECuda : public HpcOperation<CudaContext>, public T2dPDEOperation
{
private:
	CUdeviceptr d_temp_in;
	CUdeviceptr d_temp_out;

public:
	T2dPDECuda(size_t steps, size_t nrows, size_t ncols, float alpha) :
		T2dPDEOperation(steps, nrows, ncols, alpha),
		d_temp_in(0),
		d_temp_out(0)
	{}

	void AllocateHost()
	{
		T2dPDEOperation::AllocateHost();
	}

	void InitData()
	{
		T2dPDEOperation::InitData();
	}

	void VerifyResult()
	{
		T2dPDEOperation::VerifyResult();
	}

	void FreeHost()
	{
		T2dPDEOperation::FreeHost();
	}

	void Launch();
	void CopyToDevice();
	void CopyFromDevice();
	void AllocateDevice();
	void FreeDevice();
};
