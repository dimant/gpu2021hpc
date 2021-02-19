
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void childKernel_level2() 
{
	printf("Dynamic "); 
} 

__global__ void childKernel_level1() 
{
	printf("CUDA ");
	childKernel_level2<<< 1, 1 >>> ();
	// get errors and wait for child to complete 
	if (cudaSuccess != cudaGetLastError()) { return; }
	if (cudaSuccess != cudaDeviceSynchronize()) { return; }
}
__global__ void parentKernel() 
{
printf("Hello ");
// launch child 
childKernel_level1<<<1, 1 >>> ();
// get errors and wait for child to complete 
if (cudaSuccess != cudaGetLastError()) { return; }
if (cudaSuccess != cudaDeviceSynchronize()) { return; }
printf("Parallelism!\n"); 
} 

int main(int argc, char* argv[]) 
{
	// launch parent 
	parentKernel<<<1, 1 >>> ();
	if (cudaSuccess != cudaGetLastError()) { return 1; }
	// wait for parent to complete 
	if (cudaSuccess != cudaDeviceSynchronize())
	{
		return 2;
	}
	return 0;
}