#pragma once

#include <vector>
#include <string>

enum class ImgFilterType
{
    GaussianBlur7x7,
    GaussianBlur5x5,
    CompositeLaplacian,
    BasicLaplacianDiags,
    SobelEdgeX,
    Identity,
    Unknown
};

class ImgFilter
{
private:
    const ImgFilterType type;
    const float* value;

public:
    ImgFilter(ImgFilterType type) :
        type(type), value(nullptr)
    {}

    const size_t SizeX();
    const size_t SizeN();
    const float* Value();
};

void constructFiltersFromOption(const char* option, std::vector<ImgFilter*>* filters);
void parseFilterOption(const char* option, std::vector<ImgFilterType>& filterTypes);

template <class TContainer>
void split(const std::string& str, TContainer& container, char delimiter = ',')
{
    std::size_t current, previous = 0;
    current = str.find(delimiter);

    while (current != std::string::npos)
    {
        container.push_back(str.substr(previous, current - previous));
        previous = current + 1;
        current = str.find(delimiter, previous);
    }

    container.push_back(str.substr(previous, current - previous));
}
