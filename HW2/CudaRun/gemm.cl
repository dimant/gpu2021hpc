// assumes square matrices (M = K = N)
// Note: A and B are source matrices
// A is M rows by K columns
// B is K rows by N columns
// C is destination
// C is M rows by N columns
__kernel void sgemm(
	__global const float* A,
	__global const float* B,
	__global float* C,
	int widthA,
	int widthB)
{
	const int col = get_global_id(0);
	const int row = get_global_id(1);

	float sum = 0.0f;

	for (int i = 0; i < widthA; i++)
	{
		sum +=
			A[i + row * widthA] * // walking across A = src0
			B[col + i * widthB];  // walking down B = src1

	}

	if(row < widthB && col < widthA)
		C[col + row * widthB] = sum;
}