#ifndef ocl_util_h
#define ocl_util_h

#include <CL/opencl.h>

#include <string>

std::string readFile(const char* fileName);

// Sets the current working directory to be the same as the directory
// containing the running executable.
void setCwdToExeDir();

#define clChkErr(err) __clChkErr(err, __FILE__, __LINE__)
void __clChkErr(cl_uint err, const char* file, const int line);

const char* clGetErrorString(cl_int err);

#endif