#ifndef DGEMOPERATION_H
#define DGEMOPERATION_H

class DgemvOperation
{
protected:
	size_t cols;
	size_t rows;

	double alpha;
	double* h_A;
	double* h_x;
	double beta;
	double* h_y;
	double* h_z;

public:
	DgemvOperation(size_t rows, size_t cols) :
		alpha(0.0),
		h_A(nullptr), h_x(nullptr),
		beta(0.0),
		h_y(nullptr), h_z(nullptr),
		rows(rows),
		cols(cols)
	{
	}

	void AllocateHost();
	void InitData();
	void VerifyResult();
	void FreeHost();
};

#endif