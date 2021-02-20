#include "ImageOperation.h"

#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "util.h"

void ImageOperation::AllocateHost()
{
	h_input_char_image = stbi_load(input_fname, &imgCols, &imgRows, &imgChannels, 0);
	size_t size = (size_t)imgCols * (size_t)imgRows;
	h_input_float_image = new float[size];

	for (size_t i = 0; i < size; i++)
	{
		h_input_float_image[i] = (float)(int)h_input_char_image[i];
	}

	h_output_char_image = new unsigned char[size];
	h_output_float_image = new float[size];

	for (auto filter : (*filters))
	{
		total_filter_size += filter->SizeN();
	}

	h_filters = new float[total_filter_size];
	h_filter_sizes = new int[filters->size()];
}

void ImageOperation::InitData()
{
	float* cursor = h_filters;

	for (int i = 0; i < filters->size(); i++)
	{
		auto filter = filters->at(i);
		auto filterSizeX = filter->SizeX();
		auto filterSizeN = filter->SizeN();
		auto filterValue = filter->Value();

		h_filter_sizes[i] = (int)filterSizeX;

		memcpy(cursor, filterValue, filterSizeN * sizeof(float));

		cursor += filterSizeN;
	}
}

void ImageOperation::VerifyResult()
{
	size_t size = (size_t)imgCols * (size_t)imgRows;

	for (size_t i = 0; i < size; i++)
	{
		h_output_char_image[i] = (unsigned char)(unsigned int)h_output_float_image[i];
	}

	int stbErr = stbi_write_jpg(output_fname, imgCols, imgRows, imgChannels, h_output_char_image, 100);
	if (stbErr != 1)
	{
		fprintf(stderr, "Failed writing %s", output_fname);
		exit(1);
	}
}

void ImageOperation::FreeHost()
{
	free(h_input_char_image);
	delete[] h_input_float_image;
	delete[] h_output_char_image;
	delete[] h_output_float_image;
	delete[] h_filters;
	delete[] h_filter_sizes;
}
