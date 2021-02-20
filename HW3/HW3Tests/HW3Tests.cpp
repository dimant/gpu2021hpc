#include <vector>
#include "CppUnitTest.h"

#include "img_filters_impl.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HW3Tests
{
	TEST_CLASS(HW3Tests)
	{
	public:
		
		TEST_METHOD(split_should_split)
		{
			const char* option = "compositeLaplacian,basicLaplacianDiags";

			std::vector<std::string> filterTypes;

			split<std::vector<std::string>>(option, filterTypes);

			Assert::AreEqual((size_t)2, filterTypes.size());
			Assert::AreEqual(std::string("compositeLaplacian"), filterTypes[0]);
			Assert::AreEqual(std::string("basicLaplacianDiags"), filterTypes[1]);
		}

		TEST_METHOD(ParseFilterOption_should_findFilters)
		{
			const char* option = "compositeLaplacian,basicLaplacianDiags";

			std::vector<ImgFilterType> filterTypes;

			parseFilterOption(option, filterTypes);

			Assert::AreEqual((size_t)2, filterTypes.size());
			Assert::IsTrue(ImgFilterType::CompositeLaplacian == filterTypes[0]);
			Assert::IsTrue(ImgFilterType::BasicLaplacianDiags == filterTypes[1]);
		}
	};
}
