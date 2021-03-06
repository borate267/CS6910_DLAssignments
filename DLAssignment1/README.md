# DLAssignment1
This is a programming assignment in which feedforward neural networks of varying architectures have been implemented from scratch with the use of packages like numpy and pandas. The following links describe the problem given, the datasets used and how the hyperparameter sweeps were done using WANDB. 

1) [Problem Statement](https://wandb.ai/miteshk/assignments/reports/Assignment-1--VmlldzozNjk4NDE?accessToken=r7ndsh8lf4wlxyjln7phvvfb8ftvc0n4lyn4tiowdg06hhzpzfzki4jrm28wqh44)
2) [Datasets](https://github.com/zalandoresearch/fashion-mnist)
3) [Setting up sweeps](https://wandb.ai/site/articles/introduction-hyperparameter-sweeps)

## Contents

### assignment1.py 

The file contains the implementation of a feed forward NN with back propogation supported by various optimisers. The sweep features are also configured. 

### assignment1.ipynb

The same codes are also uploaded as a Jupyter notebook.

## Optimisers 

The following optimisers have been implemented:
1) Stochatic gradient Descent
2) Momentum based gradient descent
3) Nesterov Accelarated gradient descent
4) RMSprop
5) Adam
6) Nadam 

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
 
