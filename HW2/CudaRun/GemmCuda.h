#ifndef GEMMCUDA_H
#define GEMMVCUDA_H

#include <cuda.h>

#include "CudaModule.h"
#include "HpcOperation.h"
#include "GemmOperation.h"

template <class T>
class GemmCuda : public HpcOperation<CudaContext>, public GemmOperation<T>
{
private:
	using GemmOperation<T>::h_A;
	using GemmOperation<T>::h_B;
	using GemmOperation<T>::h_C;
	using GemmOperation<T>::widthA;
	using GemmOperation<T>::heightA;
	using GemmOperation<T>::widthB;
	using GemmOperation<T>::heightB;

	CUdeviceptr d_A;
	CUdeviceptr d_B;
	CUdeviceptr d_C;

public:
	GemmCuda<T>(int widthA, int heightA, int widthB, int heightB) :
		d_A(0), d_B(0), d_C(0),
		GemmOperation<T>(widthA, heightA, widthB, heightB)
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