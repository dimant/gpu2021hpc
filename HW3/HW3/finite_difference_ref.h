#pragma once
#define I2D(num, c, r) ((r)*(num)+(c))

// ni - columns
// nj - rows
// tfac - alpha
// uses 0.0f to fill missing cells
void center_diff(int nstep, int ni, int nj, float tfac, float* temp1_ref, float* temp2_ref);

// uses edge value to fill mising cells
void center_clamp_diff(int nstep, int ni, int nj, float tfac, float* temp_in, float* temp_out);

void full_diff(int nstep, int ni, int nj, float tfac, float* temp_in, float* temp_out);