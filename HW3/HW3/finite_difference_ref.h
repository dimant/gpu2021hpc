#pragma once
#define I2D(num, c, r) ((r)*(num)+(c))

// ni - columns
// nj - rows
// tfac - alpha
void center_diff(int nstep, int ni, int nj, float tfac, float* temp1_ref, float* temp2_ref);

void full_diff(int nstep, int ni, int nj, float tfac, float* temp_in, float* temp_out);