#ifndef CUDA_OPERATION_H
#define CUDA_OPERATION_H

#include <iostream>

#include <windows.h>

template <class T>
class HpcOperation
{
private:

	T context;

protected:
	void SetContext(T cuF)
	{
		context = cuF;
	}

	T GetContext()
	{
		return context;
	}

public:
	HpcOperation()
	{
	}

	void Process(T context)
	{
		double pcFreq = 0.0;
		LARGE_INTEGER li;
		__int64 counterStart = 0;
		double counterElapsed = 0;

		if (!QueryPerformanceFrequency(&li))
		{
			std::cout << "QueryPerformanceFrequency failed!\n";
			exit(EXIT_FAILURE);
		}

		pcFreq = double(li.QuadPart) / 1000.0;;

		SetContext(context);
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

	virtual void AllocateHost() = 0;
	virtual void AllocateDevice() = 0;
	virtual void InitData() = 0;
	virtual void CopyToDevice() = 0;
	virtual void Launch() = 0;
	virtual void CopyFromDevice() = 0;
	virtual void VerifyResult() = 0;
	virtual void FreeHost() = 0;
	virtual void FreeDevice() = 0;
};

#endif