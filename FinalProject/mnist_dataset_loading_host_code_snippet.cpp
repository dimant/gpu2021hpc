

// HOST CODE MNIST DATASET LOADING
// add #include "mnist_dataset_helper.h"

    // Assume MNIST dataset is in C:\FinalProject\mnist sub-folders
    string mnist_test_imgs_filepath("C:\\FinalProject\\mnist\\t10k-images-idx3-ubyte\\t10k-images.idx3-ubyte");
    string mnist_test_labels_filepath("C:\\FinalProject\\mnist\\t10k-labels-idx1-ubyte\\t10k-labels.idx1-ubyte");
    string mnist_train_imgs_filepath("C:\\FinalProject\\mnist\\train-images-idx3-ubyte\\train-images.idx3-ubyte");
    string mnist_train_labels_filepath("C:\\FinalProject\\mnist\\train-labels-idx1-ubyte\\train-labels.idx1-ubyte");

    // Functions will malloc memory for arrays, host/caller must free() it.
    // these int values will be filled with what is read from the MNIST dataset files
    int num_test_imgs = 0;
    int num_train_imgs = 0;
    int num_test_lbls = 0;
    int num_train_lbls = 0;
    int n_rows = 0;
    int n_cols = 0;
    //// loads data into one contiguous packed memory region, sample-by-sample
	// each sample label data is a one-hot-encoded vector of length = 10 where the "correct" digit = 1 and all others = 0
   float* mnist_test_imgs = NULL;
   float* mnist_test_lbls = NULL;
   float* mnist_train_imgs = NULL;
   float* mnist_train_lbls = NULL;

   load_preproc_mnist_labels(mnist_test_labels_filepath, &mnist_test_lbls, num_test_lbls); // test set
   load_preproc_mnist_labels(mnist_train_labels_filepath, &mnist_train_lbls, num_train_lbls); // train set
   load_and_preproc_mnist_images(mnist_test_imgs_filepath, &mnist_test_imgs, num_test_imgs, n_rows, n_cols); // test set
   load_and_preproc_mnist_images(mnist_train_imgs_filepath, &mnist_train_imgs, num_train_imgs, n_rows, n_cols); // train set
