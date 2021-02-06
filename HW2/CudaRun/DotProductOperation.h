#ifndef DOTPRODUCTOPERATION_H
#define DOTPRODUCTOPERATION_H

#include <cuda.h>

#include "HpcOperation.h"


template <class T>
class DotProductOperation
{
protected:
		size_t elements;
		size_t outputSize;

		T* h_A;
		T* h_B;
		float* h_C;

public:
	DotProductOperation(size_t e, size_t os) :
		elements(e), outputSize(os),
		h_A(nullptr), h_B(nullptr), h_C(nullptr)
	{
	}

	void AllocateHost();
	void InitData();
	void FreeHost();
	void VerifyResult();
};

#endif