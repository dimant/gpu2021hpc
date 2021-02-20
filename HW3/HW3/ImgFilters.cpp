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

const char* sobelEdgeX_str = "sobelEdgeX";
const size_t sobelEdgeX_x = 3;
const size_t sobelEdgeX_n = 9;
const float sobelEdgeX[sobelEdgeX_n] = {
    1.0f, 0.0f, -1.0f,
    2.0f, 0.0f, -2.0f,
    1.0f, 0.0f, -1.0f
};

const char* identity_str = "identity";
const size_t identity_x = 1;
const size_t identity_n = 1;
const float identity[identity_n] = { 1.0f };

bool starts_with(const char* str, const char* pre)
{
    size_t lenstr = strlen(str);
    size_t lenpre = strlen(pre);
    return lenstr < lenpre ? false : memcmp(pre, str, lenpre) == 0;
}

ImgFilterType filterTypeFromName(const std::string& filterName)
{
    if (starts_with(gaussianBlur7x7_str, filterName.c_str()))
    {
        return ImgFilterType::GaussianBlur7x7;
    }
    else if (starts_with(gaussianBlur5x5_str, filterName.c_str()))
    {
        return ImgFilterType::GaussianBlur5x5;
    }
    else if (starts_with(compositeLaplacian_str, filterName.c_str()))
    {
        return ImgFilterType::CompositeLaplacian;
    }
    else if (starts_with(basicLaplacianDiags_str, filterName.c_str()))
    {
        return ImgFilterType::BasicLaplacianDiags;
    }
    else if (starts_with(sobelEdgeX_str, filterName.c_str()))
    {
        return ImgFilterType::SobelEdgeX;
    }
    else if (starts_with(identity_str, filterName.c_str()))
    {
        return ImgFilterType::Identity;
    }
    else
    {
        return ImgFilterType::Unknown;
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
    case(ImgFilterType::Identity):
        return identity;
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
    case(ImgFilterType::Identity):
        return identity_x;
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
    case(ImgFilterType::Identity):
        return identity_n;
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

    if (types.size() % 2 == 0)
    {
        filters->push_back(new ImgFilter(ImgFilterType::Identity));
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
