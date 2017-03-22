__kernel void minKernel(__global const int* A, __global int* B)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N))
		{
			if (B[id] > B[id + i])
				B[id] = B[id + i];
		}		

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}
