#include <stdio.h>
#include <iostream>
#include <windows.h>

#include <CL/opencl.h>

#include <string>
#include <fstream>
#include <streambuf>


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


#define clChkErr(err) __clChkErr(err, __FILE__, __LINE__)
inline void __clChkErr(cl_uint err, const char* file, const int line)
{
    if (err != CL_SUCCESS)
    {
        std::cout << file << ":" << line << " " << clGetErrorString(err) << std::endl;
        exit(1);
    }
}

#define printClStringAttribute(attribute, device) __printClStringAttribute(attribute, #attribute, device)

inline void __printClStringAttribute(cl_device_info attribute, const char* attribute_name, cl_device_id device)
{
    char cBuffer[1024];

    clChkErr(clGetDeviceInfo(device, attribute, sizeof(cBuffer), &cBuffer, NULL));

    std::cout << attribute_name << ": " << cBuffer << std::endl;
}

#define printClIntAttribute(attribute, device) __printClIntAttribute(attribute, #attribute, device)

inline void __printClIntAttribute(cl_device_info attribute, const char* attribute_name, cl_device_id device)
{
    cl_uint value;

    clChkErr(clGetDeviceInfo(device, attribute, sizeof(cl_uint), &value, NULL));

    std::cout << attribute_name << ": " << value << std::endl;
}

void printFpConfig(cl_device_fp_config fp_config)
{
    if (fp_config & CL_FP_DENORM)
    {
        std::cout << "CL_FP_DENORM, ";
    }
    if (fp_config & CL_FP_INF_NAN)
    {
        std::cout << "CL_FP_INF_NAN, ";
    }
    if (fp_config & CL_FP_ROUND_TO_NEAREST)
    {
        std::cout << "CL_FP_ROUND_TO_NEAREST, ";
    }
    if (fp_config & CL_FP_ROUND_TO_ZERO)
    {
        std::cout << "CL_FP_ROUND_TO_ZERO, ";
    }
    if (fp_config & CL_FP_ROUND_TO_INF)
    {
        std::cout << "CL_FP_ROUND_TO_INF, ";
    }
    if (fp_config & CL_FP_FMA)
    {
        std::cout << "CL_FP_FMA, ";
    }
    std::cout << std::endl;
}

// Sets the current working directory to be the same as the directory
// containing the running executable.
bool setCwdToExeDir() {
    HMODULE hMod = GetModuleHandle(NULL);
    char path[MAX_PATH];
    GetModuleFileNameA(hMod, path, MAX_PATH);

    // Find the last '\' or '/' and terminate the path there; it is now
    // the directory containing the executable.
    size_t i;
    for (i = strlen(path) - 1; i > 0 && path[i] != '/' && path[i] != '\\'; --i);
    path[i] = '\0';

    SetCurrentDirectoryA(path);

    return true;
}

std::string readFile(const char* fileName)
{
    std::ifstream t(fileName);
    std::string str((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    return str;
}

int main(int argc, char** argv)
{
    const char* kernelFile = "vecAdd.cl";

    cl_int err;

    cl_uint num_entries = 10;
    cl_platform_id clSelectedPlatformIDs[10];
    cl_uint clNumPlatforms = 0;

    cl_device_id clDevices[10];
    cl_uint clNumDevices = 0;

    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    setCwdToExeDir();

    clChkErr(clGetPlatformIDs(num_entries, clSelectedPlatformIDs, &clNumPlatforms));

    clChkErr(clGetDeviceIDs(clSelectedPlatformIDs[0], CL_DEVICE_TYPE_ALL, num_entries, clDevices, &clNumDevices));

    // Create a context  
    context = clCreateContext(0, 1, clDevices, NULL, NULL, &err);
    clChkErr(err);

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
        clGetProgramBuildInfo(program, clDevices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char* log = (char*)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, clDevices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        std::cout << log << std::endl;
        exit(EXIT_FAILURE);
    }
    else
    {
        clChkErr(err);
    }

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vecAdd", &err); clChkErr(err);

    // Length of vectors
    unsigned int n = 100000;

    // Host input vectors
    double* h_a;
    double* h_b;
    // Host output vector
    double* h_c;

    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;
    
    size_t nbytes = n * sizeof(double);

    // Allocate memory for each vector on host
    h_a = new double[n];
    h_b = new double[n];
    h_c = new double[n];

    // Initialize vectors on host
    for (size_t i = 0; i < n; i++)
    {
        h_a[i] = i;
        h_b[i] = i;
    }

    size_t globalSize, localSize;

    // Number of work items in each local work group
    localSize = 64;

    // Number of total work items - localSize must be divisor
    globalSize = (size_t) (ceil(n / (float)localSize)) * localSize;

    // Create a command queue 
    queue = clCreateCommandQueue(context, clDevices[0], 0, &err);
    clChkErr(err);

    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &err); clChkErr(err);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, nbytes, NULL, &err); clChkErr(err);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, nbytes, NULL, &err); clChkErr(err);

    // Write our data set into the input array in device memory
    clChkErr(clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, nbytes, h_a, 0, NULL, NULL));
    clChkErr(clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, nbytes, h_b, 0, NULL, NULL));

    // Set the arguments to our compute kernel
    clChkErr(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a));
    clChkErr(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b));
    clChkErr(clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c));
    clChkErr(clSetKernelArg(kernel, 3, sizeof(unsigned int), &n));

    // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);

    // Read the results from the device
    clChkErr(clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, nbytes, h_c, 0, NULL, NULL));

    // Read the results from the device
    clChkErr(clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, nbytes, h_c, 0, NULL, NULL));

    for (int i = 0; i < n; ++i)
    {
        if (fabs(h_a[i] + h_a[i] - h_c[i]) > 1e-5)
        {
            std::cerr << "Result verification failed at element " << i << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    //release host memory
    delete [] h_a;
    delete [] h_b;
    delete [] h_c;

    return 0;
}