# DLAssignment2
Inside this repository lie two parts:
1) Part A focusses on training a CNN from scratch.
2) Part B focusses on fine-tuning parameters from a pre-existing model.

The assignment is implemented using higher level APIs like Keras. The following links describe the problem statement and the results:
1) [Problem Statement](https://wandb.ai/miteshk/assignments/reports/Assignment-2--Vmlldzo0NjA1MTU)
2) [Datasets](https://github.com/borate267/inaturalist-dataset)
3) [Setting up sweeps](https://wandb.ai/site/articles/introduction-hyperparameter-sweeps)
4) [Report](https://wandb.ai/bharatik/cs6910assignment2/reports/CS6910-Assignment-2--Vmlldzo2MDAwNjU)

## Contents
## Part A
### final_code_partA.ipynb

The above file contains the implementation of training a CNN in Keras. The sweep features configured for this are:
1) Shape of the filters ('kernel_size') : [(3,3),(3,3),(3,3),(3,3),(3,3)], [(3,3),(5,5),(5,5),(7,7),(7,7)], [(7,7),(7,7),(5,5),(5,5),(3,3)], [(3,3),(5,5),(7,7),(9,9),(11,11)] 
2) L2 regularisation ('weight_decay') : [0, 0.0005, 0.005]
3) Dropout ('dropout') : [0, 0.2, 0.4]
4) Learning rate ('learning_rate') : [1e-3, 1e-4]
5) Activation function ('activation') : ['relu', 'elu', 'selu']
6) Batch size for training ('batch_size') : 64,
7) Batch Normalisation ('batch_norm') : ['true','false']
8) Filter organisation ('filt_org' ) : [[32,32,32,32,32],[32,64,64,128,128],[128,128,64,64,32],[32,64,128,256,512]]
9) Data augmentation ('data_augment') : ['true','false']
10) Number of neurons in dense layer ('num_dense') : [64, 128, 256, 512]

### final_code_partA_testing.ipynb

This file contains the code used for testing, visualisation and guided backpropogation. The best hyperparameters that selected by the sweep are:
1) Shape of the filters ('kernel_size') : [(3,3),(3,3),(3,3),(3,3),(3,3)]
2) L2 regularisation ('weight_decay') : 0
3) Dropout ('dropout') : 0.2
4) Learning rate ('learning_rate') : 1e-4
5) Activation function ('activation') :  'selu'
6) Batch size for training ('batch_size') : 64,
7) Batch Normalisation ('batch_norm') : 'false',
8) Filter organisation ('filt_org' ) : [32,64,64,128,128]
9) Data augmentation ('data_augment') : 'false'
10) Number of neurons in dense layer ('num_dense') : 128

### guided_backprop_output.pdf
One image from each of the 10 classes is chosen. Guided backprop is applied to CONV5 and the neuron excitation of 10 neurons in that layer is visualised under each image.

## Part B
### final_code_partB.ipynb

This file contain the implementation specifics of a fine-tuning a pre-trained model in Keras. The sweep features configured for this are:

1) Freeze until before the last kth layer where k takes ['50','70',100']
2) Dropout ('dropout') : [0, 0.2, 0.4]
3) Batch size for training ('batch_size') : [32,64]
4) Number of neurons in dense layer ('num_dense') : [64, 128, 256, 512]

### final_code_partBtesting.ipynb

This file contain the puece of code used for testing on the iNaturalist dataset. The best hyperparameters are listed as follows:
1) Freeze until before the last kth layer where k takes ['100']
2) Dropout ('dropout') : [0.2]
3) Batch size for training ('batch_size') : [64]
4) Number of neurons in dense layer ('num_dense') : [512]

All the codes can be run on Jupyter notebook with the essential libraries and packages installed.

## Sweep configurations

1) Method: Bayes
2) Metric: Validation accuracy (to be maximised)
3) Stopping parameters: Early terminate (type: hyperband, min_iter = 3, s = 2)
4) Parameters (mentioned above)
 

