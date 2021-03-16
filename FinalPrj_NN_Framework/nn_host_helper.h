#pragma once

// helper routines for Neural Network (NN) final project

#pragma once
#include <string>
#include <fstream>

using namespace std;

typedef struct { int num_floats; float* pArray; float* pGradArray; } layerParamArray;

int save_network_parameters_to_binaryfile(string strSaveFileFullPath, const int& num_arrays, layerParamArray* pParamArrays)
{
	ofstream file(strSaveFileFullPath, ios::binary | ios::out);

	if (file.is_open())
	{
		for (int i = 0; i < num_arrays; ++i)
		{
			file.write((char*)pParamArrays[i].pArray, pParamArrays[i].num_floats * sizeof(float));
		}
		file.close();
	}
	else
	{
		return 1; // FAIL
	}

	return 0; // No error
}

int load_network_parameters_from_binaryfile(string strNNParamFileFullPath, const int& num_arrays, layerParamArray* pParamArrays)
{
	ifstream file(strNNParamFileFullPath, ios::binary | ios::in);

	if (file.is_open())
	{
		for (int i = 0; i < num_arrays; ++i)
		{
			file.read((char*)pParamArrays[i].pArray, pParamArrays[i].num_floats * sizeof(float));

		}
		file.close();
	}
	else
	{
		return 1; // FAIL
	}

	return 0; // NO error
}


