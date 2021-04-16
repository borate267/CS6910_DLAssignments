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

### final_code_partA.ipynb
### final_code_partA_testing.ipynb

The above two files contain the implementation of a CNN in Keras (training + testing). The sweep features configured for this are:

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

### final_code_partB.ipynb
### final_code_partBtesting.ipynb

The same codes are also uploaded as a Jupyter notebook.



## Hyperparameter tuning

The best hyperparameters are to be selected based on validation results. Following are the key parameters of interest. 

1) number of epochs: 5, 10
2) number of hidden layers: 3, 4, 5
3) size of every hidden layer: 32, 64, 128
4) weight decay (L2 regularisation): 0, 0.0005, 0.5
5) learning rate: 1e-3, 1 e-4
6) optimizer: sgd, momentum, nesterov, rmsprop, adam, nadam
7) batch size: 16, 32, 64
8) weight initialisation: random, Xavier
9) activation functions: sigmoid, tanh, ReLU

Note: There are other parameters that can be tuned but have not been experimented with.

## Sweep configurations

1) Method: Bayes
2) Metric: Validation accuracy (to be maximised)
3) Stopping parameters: Early terminate (type: hyperband, min_iter = 3, s = 2)
4) Parameters (mentioned above)
 

