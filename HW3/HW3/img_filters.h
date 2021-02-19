#pragma once

#include <vector>

enum ImgFilterType
{ 
    GaussianBlur7x7,
    GaussianBlur5x5,
    CompositeLaplacian,
    BasicLaplacianDiags,
    SobelEdgeX
};

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

std::vector<ImgFilterType> ParseFilterOption(const char* option);
