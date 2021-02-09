#ifndef UTIL_H
#define UTIL_H

#include <string>

struct Size3
{
	unsigned int x, y, z;
};

struct Work
{
	Size3 threads, blocks;
};

std::string readFile(const char* fileName);

// Sets the current working directory to be the same as the directory
// containing the running executable.
void setCwdToExeDir();

void checkTolerance(double tolerance);

#endif