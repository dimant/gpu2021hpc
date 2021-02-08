#ifndef GEMMOPERATION_H
#define GEMMOPERATION_H

class GemmOperation
{
protected:
	float* h_A;
	float* h_B;
	float* h_C;

	int widthA;
	int heightA;
	int widthB;
	int heightB;

public:
	GemmOperation(int widthA, int heightA, int widthB, int heightB) :
		h_A(nullptr), h_B(nullptr), h_C(nullptr),
		widthA(widthA), heightA(heightA),
		widthB(widthB), heightB(heightB)
	{
	}

	void AllocateHost();
	void InitData();
	void VerifyResult();
	void FreeHost();
};

#endif