#include <stdio.h>
#include <iostream>

#include "ocl_util.h"

cl_kernel oclCompileKernel(cl_device_id device, cl_context& context, cl_program& program, const char* kernelFile, const char* kernelName)
{
    cl_int err;
    cl_kernel kernel;

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

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, kernelName, &err); clChkErr(err);

    return kernel;
}

int main(int argc, char** argv)
{
    const char* kernelFile = "vecAdd.cl";
    const char* kernelName = "vecAdd";

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

    kernel = oclCompileKernel(clDevices[0], context, program, kernelFile, kernelName);

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
        h_a[i] = (double) i;
        h_b[i] = (double) i;
    }

    size_t globalSize, localSize;

    // Number of work items in each local work group
    localSize = 64;

    // Number of total work items - localSize must be divisor
    globalSize = (size_t)(ceil(n / (float)localSize)) * localSize;

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

    for (size_t i = 0; i < n; i++)
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