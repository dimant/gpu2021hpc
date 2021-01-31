#ifndef OPENCLMODULE_H
#define OPENCLMODULE_H

#include <string>

#include <CL/opencl.h>

#include "ocl_util.h"


class OpenCLModule
{
private:
	cl_device_id device;
	cl_program program;
	cl_kernel kernel;
	cl_command_queue queue;
	cl_context context;

	std::string _kernelFile;
	std::string _kernelName;

public:
	OpenCLModule() :
		device(),
		program(nullptr),
		kernel(nullptr),
		queue(nullptr),
		context(nullptr)
	{
	}

	~OpenCLModule()
	{
		clChkErr(clReleaseProgram(program));
		clChkErr(clReleaseKernel(kernel));
		clChkErr(clReleaseCommandQueue(queue));
		clChkErr(clReleaseContext(context));
	}

	void Init();

	void Compile(const char* kernelFile);

	cl_kernel GetFunction(const char* kernelName);
};

#endif