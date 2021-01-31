#ifndef MATADDOPENCL_H
#define MATADDOPENCL_H

#include <CL/opencl.h>

#include "OpenCLModule.h"
#include "HpcOperation.h"
#include "MatAddOperation.h"

class MatAddOpenCL : public HpcOperation<OpenCLContext>, public MatAddOperation
{
private:
	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

public:
	MatAddOpenCL(size_t nr, size_t nc) :
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
