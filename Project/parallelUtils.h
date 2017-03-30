#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#include <iostream>
#include <vector>
#include <memory>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace parallel_assignment
{
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

		while (workgroupSize > 1)
		{			
			parallel_assignment::Kernel myKernel("addition_reduce_unwrapped", local_size, context, queue, program);
			if (output.size() == 0)
			{
				myKernel.AddBuffer(data, true);
				workgroupSize = (data.size() / local_size) / 2;
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
			output.resize(workgroupSize);
			myKernel.ReadBuffer(outputNum, output);

			PadVector(output, local_size);
		}

		std::cout << "Output :" << output << std::endl;

		return output[0];
	}
};
