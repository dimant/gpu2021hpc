#ifndef DGEMOPERATION_H
#define DGEMOPERATION_H

#include "CudaOperation.h"

class DgemvOperation : public CudaOperation
{
private:
	size_t cols;
	size_t rows;

	double alpha;
	double* h_A;
	double* h_x;
	double beta;
	double* h_y;
	double* h_z;

	CUdeviceptr d_A;
	CUdeviceptr d_x;
	CUdeviceptr d_y;
	CUdeviceptr d_z;

public:
	DgemvOperation(size_t rows, size_t cols) :
		alpha(0.0),
		h_A(nullptr), h_x(nullptr),
		beta(0.0),
		h_y(nullptr), h_z(nullptr),
		d_A(0), d_x(0), d_y(0), d_z(0),
		rows(rows),
		cols(cols)
	{
	}

	void AllocateHost();
	void AllocateDevice();
	void InitData();
	void CopyToDevice();
	void Launch();
	void CopyFromDevice();
	void VerifyResult();
	void FreeResources();
};

#endif