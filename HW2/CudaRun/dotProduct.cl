#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void dotProductFloat(__global const float* A, __global const float* B, __global float* C, int size)
{
	__local  float sumBuffer[128];

	int sumBufferIdx = get_local_id(0);
	int stride = get_global_size(0);

	float strideSum = 0;

	for (int cellIdx = get_global_id(0); cellIdx < size; cellIdx += stride)
	{
		strideSum = strideSum + A[cellIdx] * B[cellIdx];
	}

	sumBuffer[sumBufferIdx] = strideSum;

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i != 0; i /= 2)
	{
		if (sumBufferIdx < i)
		{
			sumBuffer[sumBufferIdx] += sumBuffer[sumBufferIdx + i];
		}

		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	if (get_local_id(0) == 0)
	{
		C[get_group_id(0)] = sumBuffer[0];
	}
}

__kernel void dotProductFloat2(__global const float2* A, __global const float2* B, __global float2* C, int size)
{
	__local  float sumBuffer[128];

	int sumBufferIdx = get_local_id(0);
	int stride = get_global_size(0);

	float strideSum = 0;

	for (int cellIdx = get_global_id(0); cellIdx < size; cellIdx += stride)
	{
		strideSum = strideSum + A[cellIdx].x * B[cellIdx].x;
		strideSum = strideSum + A[cellIdx].y * B[cellIdx].y;
	}

	sumBuffer[sumBufferIdx] = strideSum;

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i != 0; i /= 2)
	{
		if (sumBufferIdx < i)
		{
			sumBuffer[sumBufferIdx] += sumBuffer[sumBufferIdx + i];
		}

		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	if (get_local_id(0) == 0)
	{
		C[get_group_id(0)] = sumBuffer[0];
	}
}

__kernel void dotProductFloat4(__global const float4* A, __global const float4* B, __global float4* C, int size)
{
	__local  float sumBuffer[128];

	int sumBufferIdx = get_local_id(0);
	int stride = get_global_size(0);

	float strideSum = 0;

	for (int cellIdx = get_global_id(0); cellIdx < size; cellIdx += stride)
	{
		strideSum = strideSum + A[cellIdx].x * B[cellIdx].x;
		strideSum = strideSum + A[cellIdx].y * B[cellIdx].y;
		strideSum = strideSum + A[cellIdx].z * B[cellIdx].z;
		strideSum = strideSum + A[cellIdx].w * B[cellIdx].w;
	}

	sumBuffer[sumBufferIdx] = strideSum;

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = get_local_size(0) / 2; i != 0; i /= 2)
	{
		if (sumBufferIdx < i)
		{
			sumBuffer[sumBufferIdx] += sumBuffer[sumBufferIdx + i];
		}

		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	if (get_local_id(0) == 0)
	{
		C[get_group_id(0)] = sumBuffer[0];
	}
}