const int threadsPerBlock = 256;

extern "C" __global__ void dotProductFloat(const float* A, const float* B, float* C, int size)
{
	__shared__ float sumBuffer[threadsPerBlock];

	int sumBufferIdx = threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	float strideSum = 0;

	for (int cellIdx = blockIdx.x * blockDim.x + threadIdx.x; cellIdx < size; cellIdx += stride)
	{
		strideSum = strideSum + A[cellIdx] * B[cellIdx];
	}

	sumBuffer[sumBufferIdx] = strideSum;

	__syncthreads();

	for (int i = blockDim.x / 2; i != 0; i /= 2)
	{
		if (sumBufferIdx < i)
		{
			sumBuffer[sumBufferIdx] += sumBuffer[sumBufferIdx + i];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		C[blockIdx.x] = sumBuffer[0];
	}
}

extern "C" __global__ void dotProductFloat2(const float2* A, const float2* B, float* C, int size)
{
	__shared__ float sumBuffer[threadsPerBlock];

	int sumBufferIdx = threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	float strideSum = 0;

	for (int cellIdx = blockIdx.x * blockDim.x + threadIdx.x; cellIdx < size; cellIdx += stride)
	{
		strideSum += A[cellIdx].x * B[cellIdx].x;
		strideSum += A[cellIdx].y * B[cellIdx].y;
	}

	sumBuffer[sumBufferIdx] = strideSum;

	__syncthreads();

	for (int i = blockDim.x / 2; i != 0; i /= 2)
	{
		if (sumBufferIdx < i)
		{
			sumBuffer[sumBufferIdx] += sumBuffer[sumBufferIdx + i];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		C[blockIdx.x] = sumBuffer[0];
	}
}

extern "C" __global__ void dotProductFloat4(const float4 * A, const float4 * B, float* C, int size)
{
	__shared__ float sumBuffer[threadsPerBlock];

	int sumBufferIdx = threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	float strideSum = 0;

	for (int cellIdx = blockIdx.x * blockDim.x + threadIdx.x; cellIdx < size; cellIdx += stride)
	{
		strideSum += A[cellIdx].x * B[cellIdx].x;
		strideSum += A[cellIdx].y * B[cellIdx].y;
		strideSum += A[cellIdx].z * B[cellIdx].z;
		strideSum += A[cellIdx].w * B[cellIdx].w;
	}

	sumBuffer[sumBufferIdx] = strideSum;

	__syncthreads();

	for (int i = blockDim.x / 2; i != 0; i /= 2)
	{
		if (sumBufferIdx < i)
		{
			sumBuffer[sumBufferIdx] += sumBuffer[sumBufferIdx + i];
		}

		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		C[blockIdx.x] = sumBuffer[0];
	}
}