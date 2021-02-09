__kernel void transpose(__global int* A, __global int* B, const int rows, const int cols)
{
	int col = get_global_id(0);
	int row = get_global_id(1);
	int index = row * rows + col;
	int transposedIndex = col * rows + row;

	if (col < cols && row < rows)
	{
		B[index] = A[transposedIndex];
	}
}
