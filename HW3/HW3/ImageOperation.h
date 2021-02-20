#ifndef IMAGEOPERATION_H
#define IMAGEOPERATION_H

class ImageOperation
{
protected:
	const char* fname;
	int imgRows;
	int imgCols;
	int imgChannels;
	unsigned char* h_input_char_image;
	float* h_input_float_image;
	unsigned char* h_output_char_image;
	float* h_output_float_image;

public:
	ImageOperation(const char* fname) :
		fname(fname),
		imgRows(0), imgCols(0), imgChannels(0),
		h_input_char_image(nullptr),
		h_input_float_image(nullptr),
		h_output_char_image(nullptr),
		h_output_float_image(nullptr)
	{
	}

	void AllocateHost();
	void InitData();
	void VerifyResult();
	void FreeHost();
};

#endif