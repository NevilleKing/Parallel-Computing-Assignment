﻿__kernel void minKernel(__global const int* A, __global int* B, __local int* scratch)
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
