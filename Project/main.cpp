#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#define ASSIGNMENT_FILENAME "temp_lincolnshire.txt"
#define LOCAL_SIZE 128

#include <iostream>
#include <vector>

#ifdef __APPLE__
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include "Utils.h"

#include <chrono>

#include "ReadFile.h"
#include "Kernel.h"
#include "parallelUtils.h"

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::high_resolution_clock::time_point TimePoint;

void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++)	{
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

	//detect any potential exceptions
	try {
		//Part 2 - host operations
		//2.1 Select computing devices
		cl::Context context = GetContext(platform_id, device_id);

		//display the selected device
		std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

		//create a queue to which we will push commands for the device
		cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);

		//2.2 Load & build the device code
		cl::Program::Sources sources;

		AddSources(sources, "kernel.cl");

		cl::Program program(context, sources);

		//build and debug the kernel code
		try {
			program.build();
		}
		catch (const cl::Error& err) {
			std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
			throw err;
		}

		typedef float mytype;

		//Part 4 - memory allocation
		//host - input

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = LOCAL_SIZE;

		TimePoint startPoint = Clock::now();

		// Read in file	
		parallel_assignment::ReadFile myFile;
		myFile.Load(ASSIGNMENT_FILENAME, local_size);

		auto timeTaken = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - startPoint).count();

		std::cout << "Read & Parse (s): " << timeTaken / 1000.f << std::endl;

		//host - output

#pragma region min_max_kernels

		int workgroupSize = (myFile.GetDataSize() / local_size) / 2;

		std::vector<mytype> minOutput(workgroupSize);

		// create an instance of the kernel class to run the min stat in parallel
		parallel_assignment::Kernel min_kernel("minimum_reduce_unwrapped", local_size, context, queue, program);
		min_kernel.AddBuffer(myFile.GetData(), true);
		int output = min_kernel.AddBuffer<mytype>(minOutput.size());
		min_kernel.AddLocalArg<mytype>();

		min_kernel.Execute();
		
		min_kernel.ReadBuffer(output, minOutput);

		TimePoint current = Clock::now();

		float minimum = minOutput[0];
		for (int i = 0; i < workgroupSize; i++)
		{
			if (minOutput[i] < minimum) minimum = minOutput[i];
		}

		TimePoint end = Clock::now();

		std::cout << "\nMinimum: " << minimum << std::endl;
		std::cout << "Minimum Time (ns): " << min_kernel.GetTime() << std::endl;
		std::cout << "Minimum Time (seq) (ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - current).count() << std::endl;

		/*std::vector<mytype> maxOutput(1);

		parallel_assignment::Kernel max_kernel("maxKernel", local_size, context, queue, program);
		max_kernel.AddBufferFromBuffer(min_kernel.GetRawBuffer(0));
		output = max_kernel.AddBuffer<mytype>(maxOutput.size());
		max_kernel.AddLocalArg<mytype>();

		max_kernel.Execute();

		max_kernel.ReadBuffer(output, maxOutput);

		std::cout << "\nMaximum: " << maxOutput[0] / 100.f << std::endl;
		std::cout << "Maximum Time (ns): " << max_kernel.GetTime() << std::endl;*/

#pragma endregion

#pragma region std_dev_kernel

		system("pause");

		float total = 0;

		// Below is a function which loops through the output and keeps calling the sum kernel until a single value is returned
		// It's not used because the below implementation is faster for the current dataset size (1.8M) but there is a speedup
		// if run on larger datasets
		//total = parallel_assignment::RecursiveKernel<float>("addition_reduce_unwrapped", local_size, context, queue, program, myFile.GetData());
		//total /= myFile.GetDataSize();

		std::vector<mytype> meanOutput(workgroupSize);

		// Calculate mean
		parallel_assignment::Kernel mean_kernel("addition_reduce_unwrapped", local_size, context, queue, program);
		mean_kernel.AddBuffer(myFile.GetData(), true);
		output = mean_kernel.AddBuffer<mytype>(meanOutput.size());
		mean_kernel.AddLocalArg<mytype>();

		mean_kernel.Execute();

		mean_kernel.ReadBuffer(output, meanOutput);

		current = Clock::now();

		for (int i = 0; i < workgroupSize; i++)
		{
			total += meanOutput[i];
		}

		end = Clock::now();

		total /= myFile.GetDataSize();

		std::cout << "\nMean: " << total << std::endl;
		std::cout << "Mean Time (ns): " << mean_kernel.GetTime() << std::endl;
		std::cout << "Mean Time (seq) (ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - current).count() << std::endl;

		
		system("pause");

		// for each number subtract mean and square result
		parallel_assignment::Kernel var_subt("variance_subtract", local_size, context, queue, program);
		var_subt.AddBufferFromBuffer(mean_kernel.GetRawBuffer(0));
		output = var_subt.AddBuffer<mytype>(myFile.GetTotalSize());
		var_subt.AddArg(total);
		var_subt.AddArg(myFile.GetDataSize());

		var_subt.Execute();

		std::vector<mytype> outputTest(myFile.GetTotalSize());
		var_subt.ReadBuffer(output, outputTest);

		// sum these up and divide by number of items
		std::vector<mytype> variance_out(workgroupSize);

		parallel_assignment::Kernel variance_kernel("addition_reduce_unwrapped", local_size, context, queue, program);
		variance_kernel.AddBufferFromBuffer(var_subt.GetRawBuffer(output));
		output = variance_kernel.AddBuffer<mytype>(variance_out.size());
		variance_kernel.AddLocalArg<mytype>();

		variance_kernel.Execute();

		variance_kernel.ReadBuffer(output, variance_out);

		current = Clock::now();

		float variance = 0;
		for (int i = 0; i < workgroupSize; i++)
		{
			variance += variance_out[i];
		}

		end = Clock::now();

		variance /= myFile.GetDataSize();

		std::cout << "\nVariance: " << variance << std::endl;
		std::cout << "Variance Time (ns): " << variance_kernel.GetTime() << std::endl;
		std::cout << "Variance Time (seq) (ns): " << std::chrono::duration_cast<std::chrono::nanoseconds>(end - current).count() << std::endl;


		// square root and return
		std::cout << "\nStandard Dev: " << sqrt(variance) << std::endl;
		



#pragma endregion

	}
	catch (cl::Error err) {
		std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
	}
	catch (...) {
		std::cerr << "ERROR" << std::endl;
	}

	system("pause");

	return 0;
}

/*
	TIMES

	addition_reduce [ns]:            660224
	addition_reduce_unwrapped [ns] :  95680 (kernel) + 18286 (seq)
	addition (atom) :                 97408
	max :                            686304
	max (atom) :                      97472
	min :                            688096
	min (atom) :                      99872
*/
