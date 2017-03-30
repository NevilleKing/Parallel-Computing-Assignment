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

//Calculate Total Reduction Kernel
__kernel void addition_reduce_unwrapped(__global const float* A, __global float* B, __local float* scratch) {

	///Refrence: http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf

	// Best to use local size of 128. Doesn''t work > 128
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int size = get_global_size(0);
	// workgroup size
	int N = get_local_size(0);
	//get group index
	int Gid = get_group_id(0);

	//Halve the number of blocks (*2) and replace single load
	int I = Gid * (N*2) + lid;

	//Gridsize to control loop to maintain coalescing
	int gridSize = N*2*get_num_groups(0);
	 
	scratch[lid] = 0;

	// Data is coalesced by using the gridsize
	// An add is completed whilst the data is being copied to scratch (first step of reduction)
	while (I < size) {scratch[lid] = (A[I] + A[I+N]); I += gridSize;}

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory
	 
	//cascading algorithm. completes reduction on remaning workgroup items based on the workgroup size

	// no loops - unwrapped the loops because they slow the kernel down

	// Complete a copy if within the workgroup boundaries
	if (N >= 128) { if (lid <64) {scratch[lid] += scratch[lid + 64];}barrier(CLK_LOCAL_MEM_FENCE);} 

	// On normal reduction, when the lid gets down to 32, most threads are inactive
	// These are only executed if needed
	if (lid < 32)
	{
	if (N >= 64) scratch[lid] += scratch[lid+32];
	if (N >= 32) scratch[lid] += scratch[lid+16];
	if (N >= 16) scratch[lid] += scratch[lid+8];
	if (N >= 8) scratch[lid] += scratch[lid+4];
	if (N >= 4) scratch[lid] += scratch[lid+2];
	if (N >= 2) scratch[lid] += scratch[lid+1];
	}
	
	//copy the cache to output array for every workgroup total value
	// There will need to be some host code to add the final values or the kernel will need to be
	// called again
	if (lid == 0) B[Gid] = scratch[0];
}

__kernel void minimum_reduce_unwrapped(__global const float* A, __global float* B, __local float* scratch) {

	int id = get_global_id(0);
	int lid = get_local_id(0);
	int size = get_global_size(0);
	int N = get_local_size(0);
	int Gid = get_group_id(0);

	int I = Gid * (N*2) + lid;

	int gridSize = N*2*get_num_groups(0);
	 
	scratch[lid] = 0;

	while (I < size) {scratch[lid] = A[I]; if (scratch[lid] > A[I+N]) scratch[lid] = A[I+N]; I += gridSize;}

	barrier(CLK_LOCAL_MEM_FENCE);
	 
	if (N >= 128) { if (lid <64) {if (scratch[lid+64] < scratch[lid]) scratch[lid] = scratch[lid+64];}barrier(CLK_LOCAL_MEM_FENCE);} 

	if (lid < 32)
	{
	if (N >= 64) if (scratch[lid+32] < scratch[lid]) scratch[lid] = scratch[lid+32];
	if (N >= 32) if (scratch[lid+16] < scratch[lid]) scratch[lid] = scratch[lid+16];
	if (N >= 16) if (scratch[lid+8] < scratch[lid]) scratch[lid] = scratch[lid+8];
	if (N >= 8) if (scratch[lid+4] < scratch[lid]) scratch[lid] = scratch[lid+4];
	if (N >= 4) if (scratch[lid+2] < scratch[lid]) scratch[lid] = scratch[lid+2];
	if (N >= 2) if (scratch[lid+1] < scratch[lid]) scratch[lid] = scratch[lid+1];
	}
	
	if (lid == 0) B[Gid] = scratch[0];
}

__kernel void maximum_reduce_unwrapped(__global const float* A, __global float* B, __local float* scratch) {

	int id = get_global_id(0);
	int lid = get_local_id(0);
	int size = get_global_size(0);
	int N = get_local_size(0);
	int Gid = get_group_id(0);

	int I = Gid * (N*2) + lid;

	int gridSize = N*2*get_num_groups(0);
	 
	scratch[lid] = 0;

	while (I < size) {scratch[lid] = A[I]; if (scratch[lid] < A[I+N]) scratch[lid] = A[I+N]; I += gridSize;}

	barrier(CLK_LOCAL_MEM_FENCE);
	 
	if (N >= 128) { if (lid <64) {if (scratch[lid+64] > scratch[lid]) scratch[lid] = scratch[lid+64];}barrier(CLK_LOCAL_MEM_FENCE);} 

	if (lid < 32)
	{
	if (N >= 64) if (scratch[lid+32] > scratch[lid]) scratch[lid] = scratch[lid+32];
	if (N >= 32) if (scratch[lid+16] > scratch[lid]) scratch[lid] = scratch[lid+16];
	if (N >= 16) if (scratch[lid+8] > scratch[lid]) scratch[lid] = scratch[lid+8];
	if (N >= 8) if (scratch[lid+4] > scratch[lid]) scratch[lid] = scratch[lid+4];
	if (N >= 4) if (scratch[lid+2] > scratch[lid]) scratch[lid] = scratch[lid+2];
	if (N >= 2) if (scratch[lid+1] > scratch[lid]) scratch[lid] = scratch[lid+1];
	}
	
	if (lid == 0) B[Gid] = scratch[0];
}

__kernel void addition_reduce2(__global const int* A, __global unsigned int* B, __local unsigned int* scratch, int dataSize)
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

	scratch[lid] = scratch[lid] / dataSize;

	barrier(CLK_LOCAL_MEM_FENCE);

	if (!lid) {
		atomic_add(&B[0], scratch[lid]);
	}
}

__kernel void variance_subtract(__global const float* input, __global float* output, float mean, int dataSize)
{ 
	int id = get_global_id(0);

	if (id < dataSize)
		output[id] = input[id] - mean;

	output[id] = (output[id] * output[id]);
}


// selection sort - ref http://www.bealto.com/gpu-sorting_parallel-selection.html
__kernel void selection_sort(__global const float *A, __global float *B)
{ 
	int id = get_global_id(0);
	int N = get_global_size(0);

	float ikey = A[id];

	// loop through all other data points to find new position of current element
	int pos = 0;
	for	(int j = 0; j < N; j++)
	{
		float jkey = A[j];
		bool smaller = (jkey < ikey) || (jkey == ikey && j < id);
		// calculate new position
		pos += (smaller)?1:0;
	}
	// place in the new position
	B[pos] = ikey;
}
