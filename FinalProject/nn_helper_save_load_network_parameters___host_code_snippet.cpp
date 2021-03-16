

// THIS CODE SNIPPET GOES INTO YOUR HOST MAIN() Routine

   //===============================================
   //====== SAVE and LOAD Network Parameters (weights and biases)
   //
   // Very basic no-frills SAVE & LOAD functionality for network parameter arrays
   // WARNING: there is very little error handling so it is easy to break (and you should probably expect a few crashes getting it working...use debugger)
   // Works for layer types: FC, CONV, pool
   // as long as each layer weights/biases are stored in a linear 1D contiguous array of floats
   
   string strNNparamfile("C:\\FinalProject\\my2layer_FC_net_w5_b3_w9_b6.bin");  // USE a unique name which you can remember which network arch it's for!
   
   int NUM_NN_PARAM_ARRAYS = 4;
   // allocate memory for the array of layerParamArray structures - which define the size and base address of each weight/bias array
   layerParamArray* pParamArray = (layerParamArray*)malloc(NUM_NN_PARAM_ARRAYS * sizeof(layerParamArray));

   // this is just an simple example with static parameter arrays - yours will be dynmically allocated arrays
   float W1[] = { 1.f, 2.f, 3.f, 4.f, 5.f };
   float b1[] = { 6.f, 7.f, 8.f };
   float W2[] = { 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f };
   float b2[] = { 6.f, 7.f, 8.f, 1.f, 2.f, 3.f };

   // you WILL need to set up this layer sizes array with the size (in # floats) for each parameter: weights and biases get separate entries.
   int layer_sizes[] = { sizeof(W1)/4,sizeof(b1)/4,sizeof(W2)/4,sizeof(b2)/4 };


   //=== Example of SAVING network parameters to a binary file
   // here you'll pass the pointer to base address of each parameter array (pParams is an array of pointers)
   float* pParams[] = { W1,b1,W2,b2 };

   for (int i = 0; i < NUM_NN_PARAM_ARRAYS; ++i)
   {
       // there is one of these for each weight and bias array, for each layer
       pParamArray[i].num_floats = layer_sizes[i];
       pParamArray[i].pArray = pParams[i];  
   }

   if (0 != save_network_parameters_to_binaryfile(strNNparamfile, NUM_NN_PARAM_ARRAYS, pParamArray))
   {
       fprintf(stderr, "Error saving network parameters to file: %s.\n", strNNparamfile.c_str());
   }

   //=== Example of LOADING network parameters from a binary file
   layerParamArray* pParamArrayLoad = (layerParamArray*)malloc(NUM_NN_PARAM_ARRAYS * sizeof(layerParamArray));
   for (int i = 0; i < NUM_NN_PARAM_ARRAYS; ++i)
   {
       pParamArrayLoad[i].num_floats = layer_sizes[i];
       pParamArrayLoad[i].pArray = (float*)malloc(layer_sizes[i] * sizeof(float));
   }

    if (0 != load_network_parameters_from_binaryfile(strNNparamfile, NUM_NN_PARAM_ARRAYS, pParamArrayLoad))
   {
       fprintf(stderr, "Error loading network parameters from file: %s.\n", strNNparamfile.c_str());
   }
   
   //==================================