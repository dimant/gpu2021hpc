#include "device_launch_parameters.h"

//========== Activation functions and first-derivatives ===============

//=====================================
// tanh - use the cuda intrinsic tanh(x)
// 8 decimal places precision

// 6 decimal places precision
__device__ __host__ __forceinline__ float tanhPrime(float f)
{
	float t = tanh(f);
	return (1 - (t * t));
}


//=====================================
// 7-8 decimal places precision
__device__ __host__ __forceinline__ float sigmoid(float f)
{
	float d = 1 + expf(-f);
	return (1 / d);
}

// 7-8 decimal places precision
__device__ __host__ __forceinline__ float sigmoidPrime(float f)
{
	float sf = sigmoid(f);
	return (sf * (1.0f - sf));
}


//===== Softmax ================================
// NOTE: Both these softmax functions only work for 10-way-classification output vectors
// they expect the input fvec10 to point to the start of a 10-float array
// targIdx will be the value of the element used in the denominator
/
/ 7 decimal places precision
__device__ __host__ __forceinline__ float softmax10(float* fvec10, int targIdx)
{
	float sum = 0;
	float targ = expf(fvec10[targIdx]);

#pragma unroll
	for (int i = 0; i < 10; ++i)
	{
		sum += expf(fvec10[i]);
	}
	return (targ / sum);
}

// 7 decimal places precision
__device__ __host__ __forceinline__ float softmax10Prime(float* fvec10, int ds_i, int dx_k)
{
	float res = 0;
	if (ds_i == dx_k)
	{
		float tmp = softmax10(fvec10, ds_i);
		res = tmp*(1 - tmp);
	}
	else
	{
		res = softmax10(fvec10, ds_i) * softmax10(fvec10, dx_k);
	}
	return res;
}



