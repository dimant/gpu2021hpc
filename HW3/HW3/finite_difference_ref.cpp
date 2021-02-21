#include <iostream>

#include "finite_difference_ref.h"

#define I2D(num, c, r) ((r)*(num)+(c))

void step_center_diff(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
    int i00, im10, ip10, i0m1, i0p1;
    float d2tdx2, d2tdy2;

    // padding edges with 0.0f
    for (int j = 0; j < nj; j++)
    {
        for (int i = 0; i < ni; i++)
        {

            i00 = I2D(ni, i, j);

            if (j == 0)
            {
                i0p1 = I2D(ni, i, j + 1);
                d2tdy2 = 0.0f - 2 * temp_in[i00] + temp_in[i0p1];
            }
            else if(j == nj - 1)
            {
                i0m1 = I2D(ni, i, j - 1);
                d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + 0.0f;
            }
            else
            {
                i0m1 = I2D(ni, i, j - 1);
                i0p1 = I2D(ni, i, j + 1);
                d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];
            }

            if (i == 0)
            {
                ip10 = I2D(ni, i + 1, j);
                d2tdx2 = 0.0f - 2 * temp_in[i00] + temp_in[ip10];
            }
            else if (i == ni - 1)
            {
                im10 = I2D(ni, i - 1, j);
                d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + 0.0f;
            }
            else
            {
                im10 = I2D(ni, i - 1, j);
                ip10 = I2D(ni, i + 1, j);
                d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
            }

            // update temperatures
            temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
        }
    }
}

void center_diff(int nstep, int ni, int nj, float tfac, float* temp_in, float* temp_out)
{
    float* left = temp_in;
    float* right = temp_out;
    float* swap = nullptr;

    // Execute the CPU-only reference version
    for (int istep = 0; istep < nstep; istep++)
    {
        step_center_diff(ni, nj, tfac, left, right);

        // swap the temperature pointers, double-buffer-esque
        swap = left;
        left = right;
        right = swap;
    }
}
