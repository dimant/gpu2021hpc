#ifndef IMAGEOPERATION_H
#define IMAGEOPERATION_H

#include <vector>

#include "img_filters.h"

class ImageOperation
{
protected:
	const char* input_fname;
	const char* output_fname;
	int imgRows;
	int imgCols;
	int imgChannels;
	unsigned char* h_input_char_image;
	float* h_input_float_image;
	unsigned char* h_output_char_image;
	float* h_output_float_image;
	float* h_filters; // the filters which will be applied pasted one after the other
		// in the order of application
	size_t total_filter_size; // if you pass 3 filters, this will be the number of
		// floats for all filters
	int* h_filter_sizes; // contains the size along X of filter at index i 
		// (sqrt of total size for a square filter)

protected:
	std::vector<ImgFilter*>* filters;

public:
	ImageOperation(const char* input_fname, const char* output_fname, 
		std::vector<ImgFilter*>* filters) :
		input_fname(input_fname),
		output_fname(output_fname),
		imgRows(0), imgCols(0), imgChannels(0),
		filters(filters),
		h_input_char_image(nullptr),
		h_input_float_image(nullptr),
		h_output_char_image(nullptr),
		h_output_float_image(nullptr),
		h_filters(nullptr),
		total_filter_size(0),
		h_filter_sizes(nullptr)
	{
	}

	void AllocateHost();
	void InitData();
	void VerifyResult();
	void FreeHost();
};

#endif