#ifndef BLUROPERATION_H
#define BLUROPERATION_H

class BlurOperation
{
protected:
	size_t rows;
	size_t cols;

	char* h_A;
	char* h_B;

public:
	BlurOperation(size_t rows, size_t cols) :
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