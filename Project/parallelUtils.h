#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <memory>
#include <chrono>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::high_resolution_clock::time_point TimePoint;

namespace parallel_assignment
{
	// Add padding to the input vector (this makes it more effiecient by resizing to a multiple of the local size)
	template <typename T>
	int PadVector(std::vector<T>& data, int localSize)
	{
		size_t padding_size = data.size() % localSize;

		//if the input vector is not a multiple of the local_size
		//insert additional neutral elements (0 for addition) so that the total will not be affected
		if (padding_size) {
			//create an extra vector with neutral values
			std::vector<int> A_ext(localSize - padding_size, 0);
			//append that extra vector to our input
			data.insert(data.end(), A_ext.begin(), A_ext.end());

			return A_ext.size();
		}

		return 0;
	}

	// Function the keeps calling the same kernel for reduction
	// The number of elements decreases to a single value
	// This is inefficient on the data size of 1.8M elements but works
	// out better larger datasets
	template <typename F>
	F RecursiveKernel(std::string kernel_name,
					  int local_size, 
					  cl::Context& context, 
					  cl::CommandQueue& queue, 
					  cl::Program& program,
					  const std::vector<F>& data)
	{	
		int workgroupSize = 2;

		std::vector<F> output;

		float timeTaken = 0.f;

		// while we have more than 1 element
		while (workgroupSize > 1)
		{			
			// run the kernel
			parallel_assignment::Kernel myKernel("addition_reduce_unwrapped", local_size, context, queue, program);
			if (output.size() == 0) // on the first iteration, copy the data passed in
			{
				myKernel.AddBuffer(data, true);
				workgroupSize = (data.size() / local_size) / 2; // /2 because the optimized kernel reduces the size by half
			}
			else
			{
				myKernel.AddBuffer(output, true);
				workgroupSize = (output.size() / local_size) / 2;
				if (workgroupSize == 0) workgroupSize = 1;
			}
			int outputNum = myKernel.AddBuffer<F>(workgroupSize);
			myKernel.AddLocalArg<F>();

			myKernel.Execute();

			output.clear();
			output.resize(workgroupSize); // resize the output to the new WG size
			myKernel.ReadBuffer(outputNum, output);

			PadVector(output, local_size); // it now needs extra padding

			timeTaken += myKernel.GetTime();
		}

		std::cout << "Output :" << output << std::endl;
		std::cout << "Time taken [ns]: " << timeTaken << std::endl;

		return output[0];
	}
};
