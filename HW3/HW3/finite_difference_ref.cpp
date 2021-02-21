#include "finite_difference_ref.h"

#define I2D(num, c, r) ((r)*(num)+(c))

void step_center_diff(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
    int i00, im10, ip10, i0m1, i0p1;
    float d2tdx2, d2tdy2;

    // loop over all points in domain (except boundary)
    for (int j = 1; j < nj - 1; j++) {
        for (int i = 1; i < ni - 1; i++) {
            // find indices into linear memory
            // for central point and neighbours
            i00 = I2D(ni, i, j);
            im10 = I2D(ni, i - 1, j);
            ip10 = I2D(ni, i + 1, j);
            i0m1 = I2D(ni, i, j - 1);
            i0p1 = I2D(ni, i, j + 1);

            // evaluate derivatives
            d2tdx2 = temp_in[im10] - 2 * temp_in[i00] + temp_in[ip10];
            d2tdy2 = temp_in[i0m1] - 2 * temp_in[i00] + temp_in[i0p1];

            // update temperatures
            temp_out[i00] = temp_in[i00] + fact * (d2tdx2 + d2tdy2);
        }
    }
}

void center_diff(int nstep, int ni, int nj, float tfac, float* temp1_ref, float* temp2_ref)
{
    float* temp_tmp;

    // Execute the CPU-only reference version
    for (int istep = 0; istep < nstep; istep++)
    {
        step_center_diff(ni, nj, tfac, temp1_ref, temp2_ref);

        // swap the temperature pointers, double-buffer-esque
        temp_tmp = temp1_ref;
        temp1_ref = temp2_ref;
        temp2_ref = temp_tmp;
    }
}
