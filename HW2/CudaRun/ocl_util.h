#ifndef ocl_util_h
#define ocl_util_h

#include <CL/opencl.h>

#define clChkErr(err) __clChkErr(err, __FILE__, __LINE__)
void __clChkErr(cl_uint err, const char* file, const int line);

const char* clGetErrorString(cl_int err);

void oclCompileProgram(
	cl_device_id device,
	cl_context& context,
	cl_program& program,
	const char* kernelFile);

#endif