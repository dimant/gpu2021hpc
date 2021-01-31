#include "OpenCLModule.h"

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

    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

    setCwdToExeDir();

    clChkErr(clGetPlatformIDs(num_entries, clSelectedPlatformIDs, &clNumPlatforms));

    clChkErr(clGetDeviceIDs(clSelectedPlatformIDs[0], CL_DEVICE_TYPE_ALL, num_entries, clDevices, &clNumDevices));

    // Create a context  
    context = clCreateContext(0, 1, clDevices, NULL, NULL, &err);
    clChkErr(err);

    device = clDevices[0];
}

void OpenCLModule::Compile(const char* kernelFile)
{
    oclCompileProgram(device, context, program, kernelFile);
}

cl_kernel OpenCLModule::GetFunction(const char* kernelName)
{
    cl_int err;

    kernel = clCreateKernel(program, kernelName, &err); clChkErr(err);

    return kernel;
}