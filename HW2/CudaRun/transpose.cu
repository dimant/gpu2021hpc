__global void matrix_transpose_naive(int* input, int* output, int N)
{
	int indexX = threadIdx.x + blockIdx.x * blockDim.x;
	int indexY = threadIdx.y + blockIdx.y * blockDim.y;
	int index = indexY * N + indexX;
	int transposedIndex = indexX * N + indexY;

	output[index] = input[transposedIndex];
}