#ifndef TRANSPOSEOPERATION_H
#define TRANSPOSEOPERATION_H

class TransposeOperation
{
protected:
	size_t cols;
	size_t rows;

	int* h_A;
	int* h_B;

public:
	TransposeOperation(size_t rows, size_t cols) :
		h_A(nullptr), h_B(nullptr),
		rows(rows), cols(cols)
	{
	}

	void AllocateHost();
	void InitData();
	void VerifyResult();
	void FreeHost();
};

#endif