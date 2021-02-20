#include <iostream>

#include "popl.h"

#include "CudaModule.h"
#include "OpenCLModule.h"

#include "img_filters.h"

#include "ImageCuda.h"

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

int main(int argc, char** argv)
{
	popl::OptionParser op("HW3 options");

	int threads;
	int blocks;

	auto help_option = op.add<popl::Switch>("h", "help", "produce help message");
	auto threads_option = op.add<popl::Value<int>>("t", "threads", "Number of x and y threads", 32, &threads);
	auto blocks_option = op.add<popl::Value<int>>("b", "blocks", "Number of x and y blocks", 64, &blocks);
	auto kernelName_option = op.add<popl::Value<std::string>>("k", "kernel-name", "Name of kernel.");
	auto filterNames_option = op.add<popl::Value<std::string>>("f", "filter-names", "Coma separated list of filters.");
	auto inputFile_option = op.add<popl::Value<std::string>>("i", "input-file", "Path to input image.");
	auto outputFile_option = op.add<popl::Value<std::string>>("o", "output-file", "Path to output image (jpg).");

	op.parse(argc, argv);

	if (!kernelName_option->is_set() || help_option->is_set())
	{
		std::cout << op << "\n";
	}
	else
	{
		std::string kernelName = kernelName_option->value();

		if (!inputFile_option->is_set() || !outputFile_option->is_set())
		{
			std::cout << op << "\n";
		}

		std::string inputFile = inputFile_option->value();
		std::string outputFile = outputFile_option->value();

		if (kernelName.rfind("imageFilter", 0) == 0)
		{
			if (!filterNames_option->is_set())
			{
				std::cout << op << "\n";
			}
			else
			{
				std::string filterNames = filterNames_option->value();
				imageFilterCuda(kernelName, filterNames, inputFile, outputFile, threads, blocks);
			}
		}
	}
}
