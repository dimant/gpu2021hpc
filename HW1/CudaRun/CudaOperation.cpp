#include <iostream>

#include <windows.h>

#include "CudaOperation.h"

void CudaOperation::Process(CUfunction cuFunction)
{
	double pcFreq = 0.0;
	LARGE_INTEGER li;
	__int64 counterStart = 0;
	__int64 counterElapsed = 0;

	if (!QueryPerformanceFrequency(&li))
	{
		std::cout << "QueryPerformanceFrequency failed!\n";
		exit(EXIT_FAILURE);
	}

	pcFreq = double(li.QuadPart) / 1000.0;;

	SetFunction(cuFunction);
	AllocateHost();
	AllocateDevice();
	InitData();
	CopyToDevice();

	if (!QueryPerformanceCounter(&li))
	{
		std::cout << "QueryPerformanceCounter failed!\n";
		exit(EXIT_FAILURE);
	}

	counterStart = li.QuadPart;

	Launch();
	CopyFromDevice();

	if (!QueryPerformanceCounter(&li))
	{
		std::cout << "QueryPerformanceCounter failed!\n";
		exit(EXIT_FAILURE);
	}

	counterElapsed = double(li.QuadPart - counterStart) / pcFreq;

	std::cout << "Execution duration: " << counterElapsed << "ms" << std::endl;

	VerifyResult();
	FreeHost();
	FreeDevice();
}