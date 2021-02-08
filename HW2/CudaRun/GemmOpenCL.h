#ifndef GEMMOPENCL_H
#define GEMMOPENCL_H

#include <CL/opencl.h>

#include "OpenCLModule.h"
#include "HpcOperation.h"
#include "GemmOperation.h"

class GemmOpenCL : public HpcOperation<OpenCLContext>, public GemmOperation
{
private:
	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

public:
	GemmOpenCL(int widthA, int heightA, int widthB, int heightB) :
		d_A(0), d_B(0), d_C(0), GemmOperation(widthA, heightA, widthB, heightB)
	{
	}

	void AllocateHost()
	{
		GemmOperation::AllocateHost();
	}

	void InitData()
	{
		GemmOperation::InitData();
	}

	void VerifyResult()
	{
		GemmOperation::VerifyResult();
	}

	void FreeHost()
	{
		GemmOperation::FreeHost();
	}

	void Launch();
	void CopyToDevice();
	void CopyFromDevice();
	void AllocateDevice();
	void FreeDevice();
};

#endif