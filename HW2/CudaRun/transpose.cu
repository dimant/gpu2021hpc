extern "C" __global__ void transpose(int* A, int* B, int rows, int cols)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int index = row * rows + col;
	int transposedIndex = col * rows + row;

	if (col < cols && row < rows)
	{
		B[index] = A[transposedIndex];
	}
}