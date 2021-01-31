#include "OpenCLModule.h"

#include <iostream>

#include "util.h"
#include "ocl_util.h"

void OpenCLModule::Init()
{
    cl_int err;
    cl_uint num_entries = 10;
    cl_platform_id clSelectedPlatformIDs[10];
    cl_uint clNumPlatforms = 0;

    cl_device_id clDevices[10];
    cl_uint clNumDevices = 0;

    setCwdToExeDir();

    clChkErr(clGetPlatformIDs(num_entries, clSelectedPlatformIDs, &clNumPlatforms));

    char cBuffer[1024];

    int selectedPlatform = 1;
    clChkErr(clGetPlatformInfo(clSelectedPlatformIDs[selectedPlatform], CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL));
    std::cout << "Platform " << selectedPlatform << ": " << cBuffer << std::endl;

    clChkErr(clGetPlatformInfo(clSelectedPlatformIDs[selectedPlatform], CL_PLATFORM_VERSION, sizeof(cBuffer), cBuffer, NULL));
    std::cout << "Version: " << cBuffer << std::endl;

    clChkErr(clGetDeviceIDs(clSelectedPlatformIDs[selectedPlatform], CL_DEVICE_TYPE_ALL, num_entries, clDevices, &clNumDevices));
    std::cout << "Devices: " << clNumDevices << std::endl;

    clChkErr(clGetDeviceIDs(clSelectedPlatformIDs[selectedPlatform], CL_DEVICE_TYPE_ALL, num_entries, clDevices, &clNumDevices));

    // Create a context  
    context = clCreateContext(0, 1, clDevices, NULL, NULL, &err);
    clChkErr(err);

    device = clDevices[0];
    queue = clCreateCommandQueue(context, device, 0, &err);
    clChkErr(err);

}

void OpenCLModule::Compile(const char* kernelFile)
{
    oclCompileProgram(device, context, program, kernelFile);
}

OpenCLContext OpenCLModule::GetContext(const char* kernelName)
{
    cl_int err;

    kernel = clCreateKernel(program, kernelName, &err); clChkErr(err);

    OpenCLContext result;

    result.context = context;
    result.queue = queue;
    result.kernel = kernel;

    return result;
}