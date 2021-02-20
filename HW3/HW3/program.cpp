#include <iostream>

#include "popl.h"

int main(int argc, char** argv)
{
	popl::OptionParser op("HW3 options");

	auto help_option = op.add<popl::Switch>("h", "help", "produce help message");

	op.parse(argc, argv);

	std::cout << op << "\n";
}
