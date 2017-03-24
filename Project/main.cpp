#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS

#define ASSIGNMENT_FILENAME "temp_lincolnshire_short.txt"

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

		typedef int mytype;

		//Part 4 - memory allocation
		//host - input

		//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
		//if the total input length is divisible by the workgroup size
		//this makes the code more efficient
		size_t local_size = 64;

		TimePoint startPoint = Clock::now();

		// Read in file	
		parallel_assignment::ReadFile myFile;
		myFile.Load(ASSIGNMENT_FILENAME, local_size);

		auto timeTaken = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - startPoint).count();

		std::cout << "Read & Parse (s): " << timeTaken / 1000.f << std::endl;

		//host - output
		std::vector<mytype> minOutput(1);

		// create an instance of the kernel class to run the min stat in parallel
		parallel_assignment::Kernel min_kernel("minKernel", local_size, context, queue, program);
		min_kernel.AddBuffer(myFile.GetData(), true);
		int output = min_kernel.AddBuffer(minOutput.size());
		min_kernel.AddLocalArg();

		min_kernel.Execute();
		
		min_kernel.ReadBuffer(output, minOutput);

		std::cout << "\nMinimum: " << minOutput[0] / 100.f << std::endl;
		std::cout << "Minimum Time (ns): " << min_kernel.GetTime() << std::endl;

		std::vector<mytype> maxOutput(1);

		parallel_assignment::Kernel max_kernel("maxKernel", local_size, context, queue, program);
		max_kernel.AddBuffer(myFile.GetData(), true);
		output = max_kernel.AddBuffer(maxOutput.size());
		max_kernel.AddLocalArg();

		max_kernel.Execute();

		max_kernel.ReadBuffer(output, maxOutput);

		std::cout << "\nMaximum: " << maxOutput[0] / 100.f << std::endl;
		std::cout << "Minimum Time (ns): " << max_kernel.GetTime() << std::endl;

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
