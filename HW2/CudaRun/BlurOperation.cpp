#include "BlurOperation.h"

#include <iostream>

#include "util.h"

void BlurOperation::AllocateHost()
{
    h_A = (char*)malloc(cols * rows * sizeof(char));
    h_B = (char*)malloc(cols * rows * sizeof(char));

    if (h_A == nullptr || h_B == nullptr)
    {
        std::cerr << "failed to allocate host vectors" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void BlurOperation::InitData()
{
    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            size_t idx = row * cols + col;
            h_A[idx] = (int)idx % blurSize;
        }
    }
}

void BlurOperation::VerifyResult()
{
    unsigned char* b = new unsigned char[rows * cols];

    for (int row = 0; row < rows; row++)
    {
        for (int col = 0; col < cols; col++)
        {
            int pixVal = 0;
            int pixels = 0;

            for (int blurRow = -blurSize; blurRow < blurSize + 1; blurRow++)
            {
                for (int blurCol = -blurSize; blurCol < blurSize + 1; blurCol++)
                {
                    int curRow = row + blurRow;
                    int curCol = col + blurCol;

                    if (curRow > -1 && curRow < rows && curCol > -1 && curCol < cols)
                    {
                        pixVal += h_A[curRow * cols + curCol];
                        pixels++;
                    }
                }
            }

            b[row * cols + col] = (unsigned char)(pixVal / pixels);
        }
    }

    bool verified = true;

    for (int i = 0; i < rows * cols; i++)
    {
        if (h_B[i] != b[i])
        {
            std::cout << "Failed verifying blur." << std::endl;
            verified = false;
            break;
        }
    }

    delete[] b;
}

void BlurOperation::FreeHost()
{
    free(h_A);
    free(h_B);
}
