#include <iostream>

#include "popl.h"

#include "CudaModule.h"
#include "OpenCLModule.h"

#include "img_filters.h"

#include "ImageCuda.h"
#include "T2dPDECuda.h"

void imageFilterCuda(std::string kernelName, std::string filterNames,
	std::string inputFile, std::string outputFile,
	int threads, int blocks)
{
	std::vector<ImgFilter*> filters;
	constructFiltersFromOption(filterNames.c_str(), &filters);

	CudaModule cudaModule;
	cudaModule.Init();

	cudaModule.Compile("imageFilter.cu");

	ImageCuda imageOperation(inputFile.c_str(), outputFile.c_str(), &filters);
	CudaContext context = cudaModule.GetContext(kernelName.c_str());
	context.work.threads.x = threads;
	context.work.threads.y = threads;
	context.work.blocks.x = blocks;
	context.work.blocks.y = blocks;

	imageOperation.Process(context);

	for (auto filter : filters)
	{
		delete filter;
	}
}

void t2PDECuda(std::string kernelName, int threads, int blocks, int steps, int nrows, int ncols, float alpha,
	const char* reference_impl)
{
	CudaModule cudaModule;
	cudaModule.Init();

	cudaModule.Compile("t2dPDE.cu");

	T2dPDECuda  t2dPDEOperation((size_t)steps, nrows, ncols, alpha, reference_impl);
	CudaContext context = cudaModule.GetContext(kernelName.c_str());
	context.work.threads.x = threads;
	context.work.threads.y = threads;
	context.work.blocks.x = blocks;
	context.work.blocks.y = blocks;

	t2dPDEOperation.Process(context);
}

int main(int argc, char** argv)
{
	popl::OptionParser op("HW3 options");

	int threads;
	int blocks;
	int nrows;
	int ncols;
	int steps;
	float alpha;

	auto help_option = op.add<popl::Switch>("h", "help", "produce help message");
	auto threads_option = op.add<popl::Value<int>>("t", "threads", "Number of x and y threads", 32, &threads);
	auto blocks_option = op.add<popl::Value<int>>("b", "blocks", "Number of x and y blocks", 64, &blocks);
	auto kernelName_option = op.add<popl::Value<std::string>>("k", "kernel-name", "Name of kernel.");
	auto filterNames_option = op.add<popl::Value<std::string>>("f", "filter-names", "Coma separated list of filters.");
	auto inputFile_option = op.add<popl::Value<std::string>>("i", "input-file", "Path to input image.");
	auto outputFile_option = op.add<popl::Value<std::string>>("o", "output-file", "Path to output image (jpg).");
	auto nrows_option = op.add<popl::Value<int>>("r", "rows", "Number of rows for test data.", 1024, &nrows);
	auto ncols_option = op.add<popl::Value<int>>("c", "cols", "Number of columns for test data.", 1024, &ncols);
	auto steps_option = op.add<popl::Value<int>>("s", "steps", "Number of steps for PDE solver.", 500, &steps);
	auto reference_impl_option = op.add<popl::Value<std::string>>("d", "reference-impl", "Reference implementation to use for verifying PDE results (center or full)", "center");

	// 8.418e-5 thermal diffusivity of silver, [m2/s]
	auto alpha_option = op.add<popl::Value<float>>("a", "alpha", "Diffusivity factor for temperature.", 8.418e-5, &alpha);

	op.parse(argc, argv);

	if (!kernelName_option->is_set() || help_option->is_set())
	{
		std::cout << op << "\n";
	}
	else
	{
		std::string kernelName = kernelName_option->value();

		if (kernelName.rfind("imageFilter", 0) == 0)
		{
			if (!inputFile_option->is_set() || !outputFile_option->is_set())
			{
				std::cout << op << "\n";
				return 0;
			}

			if (!filterNames_option->is_set())
			{
				std::cout << op << "\n";
				return 0;
			}


			std::string inputFile = inputFile_option->value();
			std::string outputFile = outputFile_option->value();

			std::string filterNames = filterNames_option->value();
			imageFilterCuda(kernelName, filterNames, inputFile, outputFile, threads, blocks);
		}
		else if (kernelName.rfind("t2dPDE", 0) == 0)
		{
			t2PDECuda(kernelName, threads, blocks, steps, nrows, ncols, alpha, reference_impl_option->value().c_str());
		}
		else
		{
			std::cout << op << "\n";
			return 0;
		}
	}
}
