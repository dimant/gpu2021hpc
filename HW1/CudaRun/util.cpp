#include <iostream>
#include <fstream>
#include <streambuf>

#include <windows.h>

#include "util.h"

std::string readFile(const char* fileName)
{
    std::ifstream t(fileName);
    std::string str((std::istreambuf_iterator<char>(t)),
        std::istreambuf_iterator<char>());

    return str;
}

void setCwdToExeDir()
{
    HMODULE hMod = GetModuleHandle(NULL);
    char path[MAX_PATH];
    GetModuleFileNameA(hMod, path, MAX_PATH);

    // Find the last '\' or '/' and terminate the path there; it is now
    // the directory containing the executable.
    size_t i;
    for (i = strlen(path) - 1; i > 0 && path[i] != '/' && path[i] != '\\'; --i);
    path[i] = '\0';

    SetCurrentDirectoryA(path);
}
