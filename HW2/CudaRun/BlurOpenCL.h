#ifndef BLUROPENCL_H
#define BLUROPENCL_H

#include <CL/opencl.h>

#include "OpenCLModule.h"
#include "HpcOperation.h"
#include "BlurOperation.h"

class BlurOpenCL : public HpcOperation<OpenCLContext>, public BlurOperation
{
private:
	cl_mem d_A;
	cl_mem d_B;

public:
	BlurOpenCL(size_t rows, size_t cols) :
		d_A(0), d_B(0), BlurOperation(rows, cols)
	{
	}

	void AllocateHost()
	{
		BlurOperation::AllocateHost();
	}

	void InitData()
	{
		BlurOperation::InitData();
	}

	void VerifyResult()
	{
		BlurOperation::VerifyResult();
	}

	void FreeHost()
	{
		BlurOperation::FreeHost();
	}

	void Launch();
	void CopyFromDevice();
	void CopyToDevice();
	void AllocateDevice();
	void FreeDevice();
};

#endif
