#include <stdio.h>

#include <cuda.h>

#ifndef checkCudaErrors
#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(CUresult err, const char* file, const int line)
{
    if (CUDA_SUCCESS != err)
    {
        const char* errorStr = NULL;
        cuGetErrorString(err, &errorStr);
        fprintf(stderr,
            "checkCudaErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            err, errorStr, file, line);
        exit(EXIT_FAILURE);
    }
}

#endif

#define printCudaIntAttribute(attribute, device) __printCudaIntAttribute(attribute, #attribute, device)

inline void __printCudaIntAttribute(CUdevice_attribute attribute, const char* attribute_name, int device)
{
    int value;
    checkCudaErrors(cuDeviceGetAttribute(&value, attribute, device));
    printf("%s: %d\n", attribute_name, value);
}

void printDriverVersion()
{
    int driverVersion = 0;
    checkCudaErrors(cuDriverGetVersion(&driverVersion));
    printf("CUDA Driver Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
}

void printDeviceTotalMemory(int device)
{
    size_t totalGlobalMem;
    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, device));

    printf("Total amount of global memory: %.0f MBytes (%llu bytes)\n",
        (float)totalGlobalMem / 1048576.0f, (unsigned long long) totalGlobalMem);
}

void printDeviceName(int device)
{
    char deviceName[256];
    checkCudaErrors(cuDeviceGetName(deviceName, 256, device));
    printf("\nDevice %d: \"%s\"\n", device, deviceName);
}

int main(int argc, char** argv)
{
    int deviceCount = 0;

    checkCudaErrors(cuInit(0));

    checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    printDriverVersion();

    for (int device = 0; device < deviceCount; device++)
    {
        printDeviceName(device);
        printDeviceTotalMemory(device);

        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y, device);
        printCudaIntAttribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z, device);
    }

    return 0;
}