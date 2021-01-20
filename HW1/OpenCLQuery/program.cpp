#include <stdio.h>
#include <iostream>

#include <CL/opencl.h>

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

inline void clChkErr(cl_uint err)
{
    if (err != CL_SUCCESS)
    {
        std::cout << clGetErrorString(err) << std::endl;
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

int main(int argc, char** argv)
{
    cl_uint num_entries = 10;
    cl_platform_id clSelectedPlatformIDs[10];
    cl_uint clNumPlatforms = 0;

    cl_device_id clDevices[10];
    cl_uint clNumDevices = 0;

    clChkErr(clGetPlatformIDs(num_entries, clSelectedPlatformIDs, &clNumPlatforms));

    char cBuffer[1024];
    cl_uint cl_uint_value;
    cl_ulong cl_ulong_value;
    cl_device_fp_config cl_device_fp_config_value;
    size_t size_t_values[1024];
    size_t results;

    for (int i = 0; i < clNumPlatforms; i++)
    {
        clChkErr(clGetPlatformInfo(clSelectedPlatformIDs[i], CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL));
        std::cout << "Platform " << i << ": " << cBuffer << std::endl;

        clChkErr(clGetPlatformInfo(clSelectedPlatformIDs[i], CL_PLATFORM_VERSION, sizeof(cBuffer), cBuffer, NULL));
        std::cout << "Version: " << cBuffer << std::endl;

        clChkErr(clGetDeviceIDs(clSelectedPlatformIDs[i], CL_DEVICE_TYPE_ALL, num_entries, clDevices, &clNumDevices));
        std::cout << "Devices: " << clNumDevices << std::endl;

        cl_int err;
        for (int j = 0; j < clNumDevices; j++)
        {
            err = clGetDeviceInfo(clDevices[j], CL_DEVICE_NAME, sizeof(cBuffer), &cBuffer, NULL);
            if (err == CL_SUCCESS)
            {
                std::cout << "CL_DEVICE_NAME: " << cBuffer << std::endl;
            }

            err = clGetDeviceInfo(clDevices[j], CL_DEVICE_EXTENSIONS, sizeof(cBuffer), &cBuffer, NULL);
            if (err == CL_SUCCESS)
            {
                std::cout << "CL_DEVICE_EXTENSIONS: " << cBuffer << std::endl;
            }
            else
            {
                std::cout << "CL_DEVICE_EXTENSIONS: no results" << std::endl;
            }

            cl_uint work_item_dimensions = 0;

            err = clGetDeviceInfo(clDevices[j], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), &cl_uint_value, NULL);
            if (err == CL_SUCCESS)
            {
                work_item_dimensions = cl_uint_value;
                std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << cl_uint_value << std::endl;
            }
            else
            {
                std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: no results" << std::endl;
            }

            err = clGetDeviceInfo(clDevices[j], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t_values), &size_t_values, &results);
            if (err == CL_SUCCESS)
            {
                std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: ";

                for (int value = 0; value < work_item_dimensions; value++)
                {
                    std::cout << size_t_values[value] << ", ";
                }

                std::cout << std::endl;
            }
            else
            {
                std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: no results" << std::endl;
            }

            err = clGetDeviceInfo(clDevices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), size_t_values, NULL);
            if (err == CL_SUCCESS)
            {
                std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: " << size_t_values[0] << std::endl;
            }
            else
            {
                std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: no results" << std::endl;
            }

            err = clGetDeviceInfo(clDevices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &cl_uint_value, NULL);
            if (err == CL_SUCCESS)
            {
                std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: " << cl_uint_value << std::endl;
            }
            else
            {
                std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: no results" << std::endl;
            }

            err = clGetDeviceInfo(clDevices[j], CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &cl_ulong_value, NULL);
            if (err == CL_SUCCESS)
            {
                std::cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: " << cl_ulong_value << std::endl;
            }
            else
            {
                std::cout << "CL_DEVICE_MAX_MEM_ALLOC_SIZE: no results" << std::endl;
            }

            err = clGetDeviceInfo(clDevices[j], CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), &cl_device_fp_config_value, NULL);
            if (err == CL_SUCCESS)
            {
                std::cout << "CL_DEVICE_SINGLE_FP_CONFIG: ";
                printFpConfig(cl_device_fp_config_value);
            }
            else
            {
                std::cout << "CL_DEVICE_SINGLE_FP_CONFIG: no results" << std::endl;
            }

            err = clGetDeviceInfo(clDevices[j], CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), &cl_device_fp_config_value, NULL);
            if (err == CL_SUCCESS)
            {
                std::cout << "CL_DEVICE_DOUBLE_FP_CONFIG: ";
                printFpConfig(cl_device_fp_config_value);
            }
            else
            {
                std::cout << "CL_DEVICE_DOUBLE_FP_CONFIG: no results" << std::endl;
            }
        }

        std::cout << std::endl;
    }

    return 0;
}