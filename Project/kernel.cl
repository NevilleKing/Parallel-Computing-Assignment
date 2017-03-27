__kernel void minKernel(__global const int* A, __global int* B, __local int* scratch)
{ 
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// copy over all data to the local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (scratch[lid] > scratch[lid + i])
				scratch[lid] = scratch[lid + i];
		}		

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_min(&B[0], scratch[lid]);
	}
}

__kernel void maxKernel(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// copy over all data to the local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
			if (scratch[lid] < scratch[lid + i])
				scratch[lid] = scratch[lid + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_max(&B[0], scratch[lid]);
	}
}

// addition kernel - used below
__kernel void addition_reduce(__global const int* A, __global unsigned int* B, __local unsigned int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	// copy over all data to the local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) { //i is a stride
		if (!(lid % (i * 2)) && ((lid + i) < N))
		{
				scratch[lid] += scratch[lid + i];
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid) {
		atomic_add(&B[0], scratch[lid]);
	}
}

__kernel void variance_subtract(__global const int* input, __global int* output, int mean, int dataSize)
{ 
	int id = get_global_id(0);

	if (id < dataSize)
		output[id] = input[id] - mean;

	output[id] = (output[id] * output[id]);
}
