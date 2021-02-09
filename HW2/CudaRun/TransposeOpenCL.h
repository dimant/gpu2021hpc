#ifndef TRANSPOSEOPENCL_H
#define TRANSPOSEOPENCL_H

#include <CL/opencl.h>

#include "OpenCLModule.h"
#include "HpcOperation.h"
#include "TransposeOperation.h"

class TransposeOpenCL : public HpcOperation<OpenCLContext>, public TransposeOperation
{
private:
	cl_mem d_A;
	cl_mem d_B;

public:
	TransposeOpenCL(size_t rows, size_t cols) :
		d_A(0), d_B(0), TransposeOperation(rows, cols)
	{
	}

	void AllocateHost()
	{
		TransposeOperation::AllocateHost();
	}

	void InitData()
	{
		TransposeOperation::InitData();
	}

	void VerifyResult()
	{
		TransposeOperation::VerifyResult();
	}

	void FreeHost()
	{
		TransposeOperation::FreeHost();
	}

	void Launch();
	void CopyFromDevice();
	void CopyToDevice();
	void AllocateDevice();
	void FreeDevice();
};

#endif
