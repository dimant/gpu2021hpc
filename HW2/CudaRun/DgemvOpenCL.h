#ifndef DGEMVOPENCL_H
#define DGEMVOPENCL_H

#include <CL/opencl.h>

#include "OpenCLModule.h"
#include "HpcOperation.h"
#include "DgemvOperation.h"

class DgemvOpenCL : public HpcOperation<OpenCLContext>, public DgemvOperation
{
private:
	cl_mem d_A;
	cl_mem d_x;
	cl_mem d_y;
	cl_mem d_z;

public:
	DgemvOpenCL(size_t rows, size_t cols) :
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