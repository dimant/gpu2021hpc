#include "T2dPDECuda.h"

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

}

void T2dPDECuda::CopyFromDevice()
{
	size_t size = (size_t)nrows * (size_t)ncols * sizeof(float);

	checkCudaError(cuMemcpyDtoH(h_temp_out, d_temp_out, size));
}

void T2dPDECuda::FreeDevice()
{
	checkCudaError(cuMemFree(d_temp_in));
	checkCudaError(cuMemFree(d_temp_in));
}
