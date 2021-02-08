#ifndef GEMMOPENCL_H
#define GEMMOPENCL_H

#include <CL/opencl.h>

#include "OpenCLModule.h"
#include "HpcOperation.h"
#include "GemmOperation.h"

template <class T>
class GemmOpenCL : public HpcOperation<OpenCLContext>, public GemmOperation<T>
{
private:
	using GemmOperation<T>::h_A;
	using GemmOperation<T>::h_B;
	using GemmOperation<T>::h_C;
	using GemmOperation<T>::widthA;
	using GemmOperation<T>::heightA;
	using GemmOperation<T>::widthB;
	using GemmOperation<T>::heightB;

	cl_mem d_A;
	cl_mem d_B;
	cl_mem d_C;

public:
	GemmOpenCL<T>(int widthA, int heightA, int widthB, int heightB) :
		d_A(0), d_B(0), d_C(0), GemmOperation<T>(widthA, heightA, widthB, heightB)
	{
	}

	void AllocateHost()
	{
		GemmOperation<T>::AllocateHost();
	}

	void InitData()
	{
		GemmOperation<T>::InitData();
	}

	void VerifyResult()
	{
		GemmOperation<T>::VerifyResult();
	}

	void FreeHost()
	{
		GemmOperation<T>::FreeHost();
	}

	void Launch();
	void CopyToDevice();
	void CopyFromDevice();
	void AllocateDevice();
	void FreeDevice();
};

#endif