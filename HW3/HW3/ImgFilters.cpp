#include <string>
#include <algorithm>
#include <iterator>

#include "img_filters.h"

const char* gaussianBlur7x7_str = "gaussianBlur7x7";
const size_t gaussianBlur7x7_x = 7;
const size_t gaussianBlur7x7_n = 49;
const float gaussianBlur7x7[gaussianBlur7x7_n] = {
   0.0086f / 3.0f,    0.0198f / 3.0f,    0.0326f / 3.0f,    0.0386f / 3.0f,    0.0326f / 3.0f,    0.0198f / 3.0f,    0.0086f / 3.0f,
   0.0198f / 3.0f,    0.0456f / 3.0f,    0.0751f / 3.0f,    0.0887f / 3.0f,    0.0751f / 3.0f,    0.0456f / 3.0f,    0.0198f / 3.0f,
   0.0326f / 3.0f,    0.0751f / 3.0f,    0.1239f / 3.0f,    0.1463f / 3.0f,    0.1239f / 3.0f,    0.0751f / 3.0f,    0.0326f / 3.0f,
   0.0386f / 3.0f,    0.0887f / 3.0f,    0.1463f / 3.0f,    0.1729f / 3.0f,    0.1463f / 3.0f,    0.0887f / 3.0f,    0.0386f / 3.0f,
   0.0326f / 3.0f,    0.0751f / 3.0f,    0.1239f / 3.0f,    0.1463f / 3.0f,    0.1239f / 3.0f,    0.0751f / 3.0f,    0.0326f / 3.0f,
   0.0198f / 3.0f,    0.0456f / 3.0f,    0.0751f / 3.0f,    0.0887f / 3.0f,    0.0751f / 3.0f,    0.0456f / 3.0f,    0.0198f / 3.0f,
   0.0086f / 3.0f,    0.0198f / 3.0f,    0.0326f / 3.0f,    0.0386f / 3.0f,    0.0326f / 3.0f,    0.0198f / 3.0f,    0.0086f / 3.0f
};

const char* gaussianBlur5x5_str = "gaussianBlur5x5";
const size_t gaussianBlur5x5_x = 5;
const size_t gaussianBlur5x5_n = 25;
const float gaussianBlur5x5[gaussianBlur5x5_n] = {
    1.0f / 273.0f, 4.0f / 273.0f, 7.0f / 273.0f, 4.0f / 273.0f, 1.0f / 273.0f,
    4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    7.0f / 273.0f, 26.0f / 273.0f, 41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f,
    4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
    1.0f / 273.0f, 4.0f / 273.0f, 7.0f / 273.0f, 4.0f / 273.0f, 1.0f / 273.0f
};

const char* compositeLaplacian_str = "compositeLaplacian";
const size_t compositeLaplacian_x = 3;
const size_t compositeLaplacian_n = 9;
const float compositeLaplacian[compositeLaplacian_n] = {
    -1.0f, -1.0f, -1.0f,
    -1.0f, 9.0f, -1.0f,
    -1.0f, -1.0f, -1.0f
};

const char* basicLaplacianDiags_str = "basicLaplacianDiags";
const size_t basicLaplacianDiags_x = 3;
const size_t basicLaplacianDiags_n = 9;
const float basicLaplacianDiags[basicLaplacianDiags_n] = {
    1.0f, 1.0f, 1.0f,
    1.0f, -8.0f, 1.0f,
    1.0f, 1.0f, 1.0f
};

const char* sobelEdgeX_str = "sobelEdgeX_str";
const size_t sobelEdgeX_x = 3;
const size_t sobelEdgeX_n = 9;
const float sobelEdgeX[sobelEdgeX_n] = {
    1.0f, 0.0f, -1.0f,
    2.0f, 0.0f, -2.0f,
    1.0f, 0.0f, -1.0f
};

ImgFilterType filterTypeFromName(const std::string& filterName)
{
    if (filterName == gaussianBlur7x7_str)
    {
        return ImgFilterType::GaussianBlur7x7;
    }
    else if (filterName == gaussianBlur5x5_str)
    {
        return ImgFilterType::GaussianBlur5x5;
    }
    else if (filterName == compositeLaplacian_str)
    {
        return ImgFilterType::CompositeLaplacian;
    }
    else if (filterName == basicLaplacianDiags_str)
    {
        return ImgFilterType::BasicLaplacianDiags;
    }
    else if (filterName == sobelEdgeX_str)
    {
        return ImgFilterType::SobelEdgeX;
    }
}

const float* filterValueFromFilterType(ImgFilterType filterType)
{
    switch (filterType)
    {
    case(ImgFilterType::GaussianBlur7x7):
        return gaussianBlur7x7;
    case(ImgFilterType::GaussianBlur5x5):
        return gaussianBlur5x5;
    case(ImgFilterType::CompositeLaplacian):
        return compositeLaplacian;
    case(ImgFilterType::BasicLaplacianDiags):
        return basicLaplacianDiags;
    case(ImgFilterType::SobelEdgeX):
        return sobelEdgeX;
    default:
        return nullptr;
    }
}

const size_t filterSizeXFromFilterType(ImgFilterType filterType)
{
    switch (filterType)
    {
    case(ImgFilterType::GaussianBlur7x7):
        return gaussianBlur7x7_x;
    case(ImgFilterType::GaussianBlur5x5):
        return gaussianBlur5x5_x;
    case(ImgFilterType::CompositeLaplacian):
        return compositeLaplacian_x;
    case(ImgFilterType::BasicLaplacianDiags):
        return basicLaplacianDiags_x;
    case(ImgFilterType::SobelEdgeX):
        return sobelEdgeX_x;
    default:
        return 0;
    }
}

const size_t filterSizeNFromFilterType(ImgFilterType filterType)
{
    switch (filterType)
    {
    case(ImgFilterType::GaussianBlur7x7):
        return gaussianBlur7x7_n;
    case(ImgFilterType::GaussianBlur5x5):
        return gaussianBlur5x5_n;
    case(ImgFilterType::CompositeLaplacian):
        return compositeLaplacian_n;
    case(ImgFilterType::BasicLaplacianDiags):
        return basicLaplacianDiags_n;
    case(ImgFilterType::SobelEdgeX):
        return sobelEdgeX_n;
    default:
        return 0;
    }
}

void parseFilterOption(const char* option, std::vector<ImgFilterType>& filterTypes)
{
    std::string str = std::string(option);
    std::vector<std::string> filterNames;

    split(str, filterNames);

    for (std::string filterName : filterNames)
    {
        filterTypes.push_back(filterTypeFromName(filterName));
    }
}

void constructFiltersFromOption(const char* option, std::vector<ImgFilter*>* filters)
{
    std::vector<ImgFilterType> types;

    parseFilterOption(option, types);

    for (auto type : types)
    {
        filters->push_back(new ImgFilter(type));
    }
}

const size_t ImgFilter::SizeX()
{
    return filterSizeXFromFilterType(type);
}

const size_t ImgFilter::SizeN()
{
    return filterSizeNFromFilterType(type);
}

const float* ImgFilter::Value()
{
    return filterValueFromFilterType(type);
}
