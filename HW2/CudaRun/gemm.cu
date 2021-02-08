// assumes square matrices (M = K = N)
// Note: A and B are source matrices
// A is M rows by K columns
// B is K rows by N columns
// C is destination
// C is M rows by N columns
extern "C" __global__ void  sgemm(
	const float* A,
	const float* B,
	float* C,
	int widthA,
	int widthB)
{
	const int col = threadIdx.x + blockIdx.x * blockDim.x;
	const int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (row < widthB && col < widthA)
	{
		float sum = 0.0f;

		for (int i = 0; i < widthA; i++)
		{
			sum += 
				A[i + row * widthA] * 
				B[col + i * widthB];
		}


		C[col + row * widthB] = sum;
	}
}