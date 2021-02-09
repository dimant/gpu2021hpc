#ifndef BLUROPERATION_H
#define BLUROPERATION_H

class BlurOperation
{
protected:
	size_t rows;
	size_t cols;
	int blurSize;

	char* h_A;
	char* h_B;

public:
	BlurOperation(size_t rows, size_t cols, unsigned int blurSize) :
		h_A(nullptr), h_B(nullptr),
		rows(rows), cols(cols), blurSize(blurSize)
	{
	}

	void AllocateHost();
	void InitData();
	void VerifyResult();
	void FreeHost();
};

#endif