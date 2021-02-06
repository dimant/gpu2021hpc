#ifndef DOTPRODUCTCUDA_H
#define DOTPRODUCTCUDA_H

#include <cuda.h>

#include "CudaModule.h"
#include "HpcOperation.h"
#include "DotProductOperation.h"

#include <device_launch_parameters.h>

template <class T>
class DotProductCuda: public HpcOperation<CudaContext>, public DotProductOperation<T>
{
private:
	using DotProductOperation<T>::elements;
	using DotProductOperation<T>::h_A;
	using DotProductOperation<T>::h_B;
	using DotProductOperation<T>::h_C;

	const int threadsPerBlock = 256;
	const int blocksPerGrid = 16;

	CUdeviceptr d_A;
	CUdeviceptr d_B;
	CUdeviceptr d_C;

public:
	DotProductCuda(size_t e) :
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