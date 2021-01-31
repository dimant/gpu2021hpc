#ifndef UTIL_H
#define UTIL_H

#include <string>

std::string readFile(const char* fileName);

// Sets the current working directory to be the same as the directory
// containing the running executable.
void setCwdToExeDir();

#endif