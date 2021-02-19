#include "ocl_util.h"

#include <iostream>
#include <fstream>
#include <streambuf>

#include "util.h"

#define clChkErr(err) __clChkErr(err, __FILE__, __LINE__)
void __clChkErr(cl_uint err, const char* file, const int line)
{
    if (err != CL_SUCCESS)
    {
        std::cout << file << ":" << line << " " << clGetErrorString(err) << std::endl;
        exit(1);
    }
}

#define CaseReturnString(x) case x: return #x;

const char* clGetErrorString(cl_int err)
{
    switch (err)
    {
        CaseReturnString(CL_SUCCESS)
            CaseReturnString(CL_DEVICE_NOT_FOUND)
            CaseReturnString(CL_DEVICE_NOT_AVAILABLE)
            CaseReturnString(CL_COMPILER_NOT_AVAILABLE)
            CaseReturnString(CL_MEM_OBJECT_ALLOCATION_FAILURE)
            CaseReturnString(CL_OUT_OF_RESOURCES)
            CaseReturnString(CL_OUT_OF_HOST_MEMORY)
            CaseReturnString(CL_PROFILING_INFO_NOT_AVAILABLE)
            CaseReturnString(CL_MEM_COPY_OVERLAP)
            CaseReturnString(CL_IMAGE_FORMAT_MISMATCH)
            CaseReturnString(CL_IMAGE_FORMAT_NOT_SUPPORTED)
            CaseReturnString(CL_BUILD_PROGRAM_FAILURE)
            CaseReturnString(CL_MAP_FAILURE)
            CaseReturnString(CL_MISALIGNED_SUB_BUFFER_OFFSET)
            CaseReturnString(CL_COMPILE_PROGRAM_FAILURE)
            CaseReturnString(CL_LINKER_NOT_AVAILABLE)
            CaseReturnString(CL_LINK_PROGRAM_FAILURE)
            CaseReturnString(CL_DEVICE_PARTITION_FAILED)
            CaseReturnString(CL_KERNEL_ARG_INFO_NOT_AVAILABLE)
            CaseReturnString(CL_INVALID_VALUE)
            CaseReturnString(CL_INVALID_DEVICE_TYPE)
            CaseReturnString(CL_INVALID_PLATFORM)
            CaseReturnString(CL_INVALID_DEVICE)
            CaseReturnString(CL_INVALID_CONTEXT)
            CaseReturnString(CL_INVALID_QUEUE_PROPERTIES)
            CaseReturnString(CL_INVALID_COMMAND_QUEUE)
            CaseReturnString(CL_INVALID_HOST_PTR)
            CaseReturnString(CL_INVALID_MEM_OBJECT)
            CaseReturnString(CL_INVALID_IMAGE_FORMAT_DESCRIPTOR)
            CaseReturnString(CL_INVALID_IMAGE_SIZE)
            CaseReturnString(CL_INVALID_SAMPLER)
            CaseReturnString(CL_INVALID_BINARY)
            CaseReturnString(CL_INVALID_BUILD_OPTIONS)
            CaseReturnString(CL_INVALID_PROGRAM)
            CaseReturnString(CL_INVALID_PROGRAM_EXECUTABLE)
            CaseReturnString(CL_INVALID_KERNEL_NAME)
            CaseReturnString(CL_INVALID_KERNEL_DEFINITION)
            CaseReturnString(CL_INVALID_KERNEL)
            CaseReturnString(CL_INVALID_ARG_INDEX)
            CaseReturnString(CL_INVALID_ARG_VALUE)
            CaseReturnString(CL_INVALID_ARG_SIZE)
            CaseReturnString(CL_INVALID_KERNEL_ARGS)
            CaseReturnString(CL_INVALID_WORK_DIMENSION)
            CaseReturnString(CL_INVALID_WORK_GROUP_SIZE)
            CaseReturnString(CL_INVALID_WORK_ITEM_SIZE)
            CaseReturnString(CL_INVALID_GLOBAL_OFFSET)
            CaseReturnString(CL_INVALID_EVENT_WAIT_LIST)
            CaseReturnString(CL_INVALID_EVENT)
            CaseReturnString(CL_INVALID_OPERATION)
            CaseReturnString(CL_INVALID_GL_OBJECT)
            CaseReturnString(CL_INVALID_BUFFER_SIZE)
            CaseReturnString(CL_INVALID_MIP_LEVEL)
            CaseReturnString(CL_INVALID_GLOBAL_WORK_SIZE)
            CaseReturnString(CL_INVALID_PROPERTY)
            CaseReturnString(CL_INVALID_IMAGE_DESCRIPTOR)
            CaseReturnString(CL_INVALID_COMPILER_OPTIONS)
            CaseReturnString(CL_INVALID_LINKER_OPTIONS)
            CaseReturnString(CL_INVALID_DEVICE_PARTITION_COUNT)
    default: return "Unknown OpenCL error code";
    }
}

void oclCompileProgram(
    cl_device_id device,
    cl_context& context,
    cl_program& program,
    const char* kernelFile)
{
    cl_int err;

    std::string kernelSource = readFile(kernelFile);
    const char* kernelString = kernelSource.c_str();

    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char**)&kernelString, NULL, &err);
    clChkErr(err);

    // Build the program executable 
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char* log = (char*)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        std::cout << log << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        clChkErr(err);
    }
}