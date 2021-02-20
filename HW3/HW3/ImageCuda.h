#pragma once

#include <cuda.h>

#include "CudaModule.h"
#include "HpcOperation.h"
#include "ImageOperation.h"

class ImageCuda : public HpcOperation<CudaContext>, public ImageOperation
{
private:
	CUdeviceptr d_input_image;
	CUdeviceptr d_output_image;
	CUdeviceptr d_filter;

public:
	ImageCuda(const char* input_fname, const char* output_fname,
		std::vector<ImgFilter*>* filters) :
		ImageOperation(input_fname, output_fname, filters),
		d_input_image(0),
		d_output_image(0),
		d_filter(0)
	{
	}

	void AllocateHost()
	{
		ImageOperation::AllocateHost();
	}

	void InitData()
	{
		ImageOperation::InitData();
	}

	void VerifyResult()
	{
		ImageOperation::VerifyResult();
	}

	void FreeHost()
	{
		ImageOperation::FreeHost();
	}

	void Launch();
	void CopyToDevice();
	void CopyFromDevice();
	void AllocateDevice();
	void FreeDevice();
};
