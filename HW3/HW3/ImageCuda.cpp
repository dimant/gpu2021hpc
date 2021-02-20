#pragma once

#include "ImageCuda.h"
#include <device_launch_parameters.h>

void ImageCuda::AllocateDevice()
{
	size_t size = (size_t) imgRows * (size_t) imgCols * sizeof(float);
	size_t filtersize = filters->at(0)->SizeN() * sizeof(float);

	checkCudaError(cuMemAlloc(&d_input_image, size));
	checkCudaError(cuMemAlloc(&d_output_image, size));
	checkCudaError(cuMemAlloc(&d_filter, filtersize));
}

void ImageCuda::CopyToDevice()
{
	size_t size = (size_t)imgRows * (size_t)imgCols * sizeof(float);

	checkCudaError(cuMemcpyHtoD(d_input_image, h_input_float_image, size));

	float* filter = const_cast<float*>(filters->at(0)->Value());
	size_t filtersize = filters->at(0)->SizeN() * sizeof(float);

	checkCudaError(cuMemcpyHtoD(d_filter, filter, filtersize));
}

void ImageCuda::Launch()
{
	dim3 blockSize(GetContext().work.threads.x, GetContext().work.threads.y);
	dim3 gridSize(GetContext().work.blocks.x, GetContext().work.blocks.y);

	int offset = 0;
	char argBuffer[1024];

	ALIGN_UP(offset, __alignof(CUdeviceptr));
	memcpy(argBuffer + offset, &(d_output_image), sizeof(d_output_image));
	offset += sizeof(d_output_image);

	ALIGN_UP(offset, __alignof(CUdeviceptr));
	memcpy(argBuffer + offset, &(d_input_image), sizeof(d_input_image));
	offset += sizeof(d_input_image);

	ALIGN_UP(offset, __alignof(CUdeviceptr));
	memcpy(argBuffer + offset, &(d_filter), sizeof(d_filter));
	offset += sizeof(d_filter);

	ALIGN_UP(offset, __alignof(int));
	memcpy(argBuffer + offset, &(imgRows), sizeof(imgRows));
	offset += sizeof(imgRows);

	ALIGN_UP(offset, __alignof(int));
	memcpy(argBuffer + offset, &(imgCols), sizeof(imgCols));
	offset += sizeof(imgCols);

	int filterSize = filters->at(0)->SizeN();
	ALIGN_UP(offset, __alignof(int));
	memcpy(argBuffer + offset, &(filterSize), sizeof(filterSize));
	offset += sizeof(filterSize);

	void* config[5] = { CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
								 CU_LAUNCH_PARAM_BUFFER_SIZE, &offset,
								 CU_LAUNCH_PARAM_END };

	checkCudaError(cuLaunchKernel(GetContext().cuFunction,
		gridSize.x, gridSize.y, gridSize.z,
		blockSize.x, blockSize.y, blockSize.z,
		0, NULL, NULL, reinterpret_cast<void**>(&config)));
}

void ImageCuda::CopyFromDevice()
{
	size_t size = (size_t)imgRows * (size_t)imgCols * sizeof(float);

	checkCudaError(cuMemcpyDtoH(h_output_float_image, d_output_image, size));
}

void ImageCuda::FreeDevice()
{
	checkCudaError(cuMemFree(d_input_image));
	checkCudaError(cuMemFree(d_output_image));
	checkCudaError(cuMemFree(d_filter));
}
