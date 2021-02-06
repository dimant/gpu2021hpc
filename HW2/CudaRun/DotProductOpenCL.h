#ifndef DOTPRODUCTOPENCL_H
#define DOTPRODUCTOPENCL_H

#include <CL/opencl.h>

#include "OpenCLModule.h"
#include "HpcOperation.h"
#include "DotProductOperation.h"

template <class T>
class DotProductOpenCL : public HpcOperation<OpenCLContext>, public DotProductOperation<T>
{
private:
	using DotProductOperation<T>::elements;
	using DotProductOperation<T>::h_A;
	using DotProductOperation<T>::h_B;
	using DotProductOperation<T>::h_C;

	const size_t localSize = 16;
	const size_t numGroups = 8;

	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

public:
	DotProductOpenCL(size_t e) :
		d_A(0), d_B(0), d_C(0), DotProductOperation<T>(e, 16)
	{
	}

	void AllocateHost();
	void InitData();
	void FreeHost();
	void VerifyResult();

	void Launch();
	void AllocateDevice();
	void CopyToDevice();
	void CopyFromDevice();
	void FreeDevice();
};

#endif