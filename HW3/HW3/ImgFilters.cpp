#include <string>
#include <algorithm>
#include <iterator>

#include "img_filters.h"

void parseFilterOption(const char* option, std::vector<ImgFilterType>& filterTypes)
{
    std::string str = std::string(option);
    std::vector<std::string> filterNames;

    split(str, filterNames);

    for (std::string filterName : filterNames)
    {
        if (filterName == gaussianBlur7x7_str)
        {
            filterTypes.push_back(ImgFilterType::GaussianBlur7x7);
        }
        else if (filterName == gaussianBlur5x5_str)
        {
            filterTypes.push_back(ImgFilterType::GaussianBlur5x5);
        }
        else if (filterName == compositeLaplacian_str)
        {
            filterTypes.push_back(ImgFilterType::CompositeLaplacian);
        }
        else if (filterName == basicLaplacianDiags_str)
        {
            filterTypes.push_back(ImgFilterType::BasicLaplacianDiags);
        }
        else if (filterName == sobelEdgeX_str)
        {
            filterTypes.push_back(ImgFilterType::SobelEdgeX);
        }
    }
}
