#include "T2dPDECuda.h"

#include <device_launch_parameters.h>

void T2dPDECuda::AllocateDevice()
{
	size_t size = (size_t)nrows * (size_t)ncols * sizeof(float);

	checkCudaError(cuMemAlloc(&d_temp_in, size));
	checkCudaError(cuMemAlloc(&d_temp_out, size));
}

void T2dPDECuda::CopyToDevice()
{
	size_t size = (size_t)nrows * (size_t)ncols * sizeof(float);

	checkCudaError(cuMemcpyHtoD(d_temp_in, h_temp_in, size));
	checkCudaError(cuMemcpyHtoD(d_temp_out, h_temp_out, size));
}

void T2dPDECuda::Launch()
{
	dim3 blockSize(1);
	dim3 gridSize(1);

	int offset = 0;
	char argBuffer[1024];

	int ithreads = GetContext().work.threads.x;
	ALIGN_UP(offset, __alignof(int));
	memcpy(argBuffer + offset, &(ithreads), sizeof(ithreads));
	offset += sizeof(ithreads);

	int iblocks = GetContext().work.blocks.x;
	ALIGN_UP(offset, __alignof(int));
	memcpy(argBuffer + offset, &(iblocks), sizeof(iblocks));
	offset += sizeof(iblocks);

	int isteps = (int)steps;
	ALIGN_UP(offset, __alignof(int));
	memcpy(argBuffer + offset, &(isteps), sizeof(isteps));
	offset += sizeof(isteps);

	int incols = (int)ncols;
	ALIGN_UP(offset, __alignof(int));
	memcpy(argBuffer + offset, &(incols), sizeof(incols));
	offset += sizeof(incols);

	int inrows = (int)nrows;
	ALIGN_UP(offset, __alignof(int));
	memcpy(argBuffer + offset, &(inrows), sizeof(inrows));
	offset += sizeof(inrows);

	ALIGN_UP(offset, __alignof(float));
	memcpy(argBuffer + offset, &(alpha), sizeof(alpha));
	offset += sizeof(alpha);

	ALIGN_UP(offset, __alignof(CUdeviceptr));
	memcpy(argBuffer + offset, &(d_temp_in), sizeof(d_temp_in));
	offset += sizeof(d_temp_in);

	ALIGN_UP(offset, __alignof(CUdeviceptr));
	memcpy(argBuffer + offset, &(d_temp_out), sizeof(d_temp_out));
	offset += sizeof(d_temp_out);

	void* config[5] = { CU_LAUNCH_PARAM_BUFFER_POINTER, argBuffer,
							 CU_LAUNCH_PARAM_BUFFER_SIZE, &offset,
							 CU_LAUNCH_PARAM_END };

	checkCudaError(cuLaunchKernel(GetContext().cuFunction,
		gridSize.x, gridSize.y, gridSize.z,
		blockSize.x, blockSize.y, blockSize.z,
		0, NULL, NULL, reinterpret_cast<void**>(&config)));
}

void T2dPDECuda::CopyFromDevice()
{
	size_t size = (size_t)nrows * (size_t)ncols * sizeof(float);

	/// <summary>
	/// since we swap input and output at every iteration, we need
	/// to be a little careful about getting the actual result and not the n-1 step.
	/// For example for steps == 2:
	///  iteration 1
	///    input  - d_temp_in
	///    output - d_temp_out
	///  iteration 2
	///    input  - d_temp_out
	///    output - d_temp_in
	/// 
	/// hence for any even number of steps, the output will be in d_temp_in but
	/// for odd number steps, the output will be in d_temp_out
	/// </summary>
	if (steps % 2 == 0)
	{
		checkCudaError(cuMemcpyDtoH(h_temp_out, d_temp_in, size));
	}
	else
	{
		checkCudaError(cuMemcpyDtoH(h_temp_out, d_temp_out, size));
	}
}

void T2dPDECuda::FreeDevice()
{
	checkCudaError(cuMemFree(d_temp_in));
	checkCudaError(cuMemFree(d_temp_out));
}
