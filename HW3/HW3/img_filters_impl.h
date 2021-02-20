#pragma once

#include <vector>
#include <string>

enum class ImgFilterType
{
    GaussianBlur7x7,
    GaussianBlur5x5,
    CompositeLaplacian,
    BasicLaplacianDiags,
    SobelEdgeX
};

class ImgFilter
{
private:
    const size_t size_x;
    const size_t size_n;
    const char* name;
    const float* value;

public:
    ImgFilter(size_t size_x, size_t size_n, const char* name, const float* value) :
        size_x(size_x), size_n(size_n), name(name), value(value)
    {}

    const size_t SizeX() { return size_x; }
    const size_t SizeN() { return size_n; }
    const char* Name() { return name; }
    const float* Value() { return value; }
};

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