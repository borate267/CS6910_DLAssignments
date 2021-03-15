# -*- coding: utf-8 -*-
"""code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FEe1LliniqGNVMOo6gZeaxRuir2djUWt
"""

# Commented out IPython magic to ensure Python compatibility.
""" The following code implements gradient descent and its variants with backpropogation for an image classification problem

Created by Bharati. K EE20D700
"""

# WandB – Install the W&B library
# %pip install wandb -q
import wandb
from wandb.keras import WandbCallback

# Essentials
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import pylab as pl

#Fetch the dataset and visualise the images
from keras.datasets import fashion_mnist

wandb.init(project="cs6910assignment1", entity="bharatik")

#Define the labels for the images that are being detected
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load the training data 
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Split data for cross validation
x_val = x_train[54000:]
y_val = y_train[54000:]
x_train = x_train[:54000]
y_train = y_train[:54000]  

image = [];
label = [];
for i in range(54000):
  if len(label) >= 10:
    break;
  if class_names[y_train[i]] not in label:
      image.append(x_train[i])
      label.append(class_names[y_train[i]])
#wandb.log({"Examples": [ wandb.Image(img, caption=caption) for img, caption in zip(image,label)]})

# Vectorise and normalize the data
x_train = x_train.reshape(x_train.shape[0], 784)
x_val  = x_val.reshape(x_val.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

x_train = x_train / 255.0
x_test = x_test / 255.0
x_val  = x_val / 255.0

# One hot encoding for labels
y_train = to_categorical(y_train)
y_val   = to_categorical(y_val)
y_test = to_categorical(y_test)

# Defining the Neural network and optimisers

# The following function initialises the weights and biases for a neural network
# that has (number_hidden + 1) layers.
def init_network( number_hidden, act_func, weight_init, hid_input_size, net_input_size, net_output_size ):
    size = [] # this contains the list of number of nodes of every layer in the network
    for j in range(number_hidden):
      size.append(hid_input_size)
    size = [net_input_size] + size + [net_output_size]
    #print(size)
    theta0 = {}

    # Reference: (1) https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
    #            (2) https://towardsdatascience.com/weight-initialization-techniques-in-neural-networks-26c649eb3b78

    if weight_init == 'xavier':
      if act_func != 'relu': #xavier weight initialisation for sigmoid and tanh
        for i in range(1, number_hidden+2):
          lower = -1/ np.sqrt(size[i-1])
          upper = 1/ np.sqrt(size[i-1])
          theta0["W" + str(i)] = lower + np.random.randn(size[i],size[i-1])*(upper- lower)
          theta0["b" + str(i)] = lower + np.random.randn(size[i],1)*(upper - lower)

      else: #He Weight Initialization for reLU activation
      # generate random number and multiply by a standard deviation that is equal to sqrt(2/number of nodes in previous layer)
        for i in range(1, number_hidden+2): 
          theta0["W" + str(i)] = np.random.randn(size[i], size[i-1])*(np.sqrt(2/(size[i-1])))
          theta0["b" + str(i)] = np.random.randn(size[i], 1)*(np.sqrt(2/(size[i-1])))

    else: #random initialisation
       for i in range(1, number_hidden+2):
          if act_func != 'relu':
            theta0["W" + str(i)] = np.random.randn(size[i], size[i-1])
            theta0["b" + str(i)] = np.random.randn(size[i], 1)
          else:
            theta0["W" + str(i)] = np.random.randn(size[i], size[i-1])*(np.sqrt(2/(size[i]+size[i-1])))
            theta0["b" + str(i)] = np.zeros((size[i],1))

    return theta0

# The following functions are used for activation at the hidden layers
def sigmoid(x):
	  return 1/(1 + np.exp(-x))

def relu(x):
    return (x>0)*(x) #+ 0.01*((x<0)*x)

def tanh(x):
	  return np.tanh(x)
   
def activation(x, act_func):
    if act_func == "sigmoid":
        return sigmoid(x)
    elif act_func == "tanh":
        return tanh(x)
    elif act_func == "relu":
        return relu(x)

# This function is used for activation at the output layer
def softmax(x):
    x = x - np.max(x)
    return np.exp(x)/np.sum(np.exp(x),axis=0)

# The following functions return the gradients of the various activation functions 
def grad_sigmoid(x):
    return (sigmoid(x))*(1 - sigmoid(x))

def grad_relu(x): # unit step
    return (x>0)*(np.ones(np.shape(x))) #+ (x<0)*(0.01*np.ones(np.shape(x)))

def grad_tanh(x):
    return (1 - (tanh(x))**2)

# The following functions are used to calculate loss
def squared_loss(x, y, sum_norm, regpara):
    loss = np.mean((y-x)**2) + regpara/2*(sum_norm)
    return loss

def cross_entropy_loss(X, Y, sum_norm, regpara):
    x = np.array(X).reshape(-1)
    y = np.array(Y).reshape(-1)
    logx = np.log(x)
    loss_vec = (-1)*(y*logx)
    loss = np.sum(loss_vec) + regpara/2*(sum_norm)
    return loss    


# Forward Propogation
def forward_propogation(x, act_func, theta, number_hidden):
    a = {} # dictionary of pre-activation values
    h = {} # dictionary of activation values
    # converting to a 2-dimensional array
    if x.ndim == 1:
      x = np.expand_dims(x, axis=1)
    h = {"h0": x} 

    for i in range(1, number_hidden + 2):
      
      if i < number_hidden + 1:

        # computing pre-activation values
        W_current = theta["W" + str(i)]
        b_current = theta["b" + str(i)]
        h_previous = h["h" + str(i-1)]
        a_current = np.dot(W_current,h_previous) + b_current
        a["a" + str(i)] = a_current

        # computing activation values
        h_current = activation(a_current, act_func)
        h["h" + str(i)] = h_current  

      else: # for output layer

        a_current = np.dot(theta["W" + str(i)],h["h" + str(i-1)]) + theta["b" + str(i)]
        h_current = softmax(a_current)  
        h["h" + str(i)] = h_current  

    y_hat = h["h" + str(number_hidden + 1)]

    return h, a, y_hat


# Initialising the gradients for backpropagation. The gradients are computed wrt
# output units, hidden layers, weights and biases
def init_grad(number_hidden, hid_input_size, net_input_size, net_output_size ):
    size = [] # this contains the list of number of nodes of every layer in the network
    for j in range(number_hidden):
        size.append(hid_input_size)
    size = [net_input_size] + size + [net_output_size]

    grad = {} 
    grad = { "dh0": np.zeros((net_input_size,1)), "da0": np.zeros((net_input_size,1)) } # is used for back prop computation
    
    for i in range(1, number_hidden + 2):
        grad["da" + str(i)] = np.zeros((size[i],1))
        grad["dh" + str(i)] = np.zeros((size[i],1))
        grad["dW" + str(i)] = np.zeros((size[i],size[i-1]))
        grad["db" + str(i)] = np.zeros((size[i],1))

    return grad

# The following function implements Backpropogation as described in Lecture 4

def backward_propogation(h, a, y, y_hat, theta, number_hidden, hid_input_size, net_input_size, net_output_size, loss_func, act_func, regpara):

    grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size ) 
    
    a["a0"] = np.zeros((net_input_size,1)) 
    
    # Computing gradient wrt output layer
    if y.ndim == 1:
      y = np.expand_dims(y, axis=1) # Converting to a 2-dimensional array
      
    if loss_func == "cross":
        grad["da" + str(number_hidden + 1)] = -1*(y - y_hat) #Enctrain label is one hot encoded
    else:
        grad["da" + str(number_hidden + 1)] = -1*(y - y_hat)*y_hat - y_hat*(np.dot((y_hat - y).T, y_hat))
    
    for i in range(number_hidden + 1 ,0,-1): # L to 1

      # Computing gradients wrt parameters
      grad["dW" + str(i)] = np.dot( grad["da" + str(i)], (h["h" + str(i-1)]).T ) + regpara*(theta["W" + str(i)])
      grad["db" + str(i)] = grad["da" + str(i)]

      # Computing gradients wrt layer below
      grad["dh" + str(i-1)] = np.dot( (theta["W" + str(i)]).T, grad["da" + str(i)] )

      # Computing gradients wrt layer below (preactivation)
      if act_func == "sigmoid":
          grad["da" + str(i-1)] = (grad["dh" + str(i-1)])*(grad_sigmoid(a["a" + str(i-1)]))      
      elif act_func == "tanh":
          grad["da" + str(i-1)] = (grad["dh" + str(i-1)])*(grad_tanh(a["a" + str(i-1)]))
      elif act_func == "relu":
          grad["da" + str(i-1)] = (grad["dh" + str(i-1)])*(grad_relu(a["a" + str(i-1)]))

    return grad


# The function computes accuracies and errors for training and validation

def loss_accuracy_compute(ind, num_epochs, max_accuracy, x_train, y_train, x_val, y_val, theta, number_hidden, act_func, loss_func, regpara):

    sum_norm = 0
    for i in range(1,number_hidden + 2):
        sum_norm = sum_norm + np.sum(np.square(theta["W"+str(i)]))

    train_loss = 0 # training loss
    val_loss = 0 # validation loss
    ctr = 0
    ctr1 = 0

    for j in range(0,54000):
        x = x_train[j,:]
        y = y_train[j,:]
        h,a,y_hat = forward_propogation(x, act_func, theta, number_hidden)
        
      
        if loss_func == "cross":
            train_loss = train_loss + (cross_entropy_loss(y_hat,y,sum_norm, regpara))
        else:
            train_loss = train_loss + (squared_loss(y, y_hat,sum_norm, regpara))
        
        
        # convert one hot encoded vector to label 
        y = np.argmax(y, axis = 0)
        y_hat = np.argmax(y_hat, axis = 0)

        if y == y_hat:
          ctr = ctr + 1

        if j < 6000:
            x_vals = x_val[j,:]
            y_vals = y_val[j,:]
            h,a,y_hat_val = forward_propogation(x_vals, act_func, theta, number_hidden)
          
            if loss_func == "cross":
                val_loss = val_loss + (cross_entropy_loss(y_hat_val,y_vals,sum_norm, regpara))
            else:
                val_loss = val_loss + (squared_loss(y_vals, y_hat_val,sum_norm, regpara))

            # convert one hot encoded vector to label 
            y_vals = np.argmax(y_vals, axis = 0)
            y_hat_val = np.argmax(y_hat_val, axis = 0)

            if y_vals == y_hat_val:
              ctr1 = ctr1 + 1


    train_acc = ctr/54000.0
    train_loss = train_loss/54000.0
    val_acc = ctr1/6000.0
    val_loss = val_loss/6000.0
    max_accuracy.append([val_acc])

    #wandb.log({"train_acc": train_acc, "train_loss": train_loss, "val_acc": val_acc, "val_loss":val_loss})
    
    """
    if (ind + 1) == num_epochs:
      max_acc = np.max(max_accuracy)
      wandb.log({"accuracy": max_acc })
      """
    return val_acc, val_loss, train_acc, train_loss

    

# Vanilla Gradient Descent Algorithm (Lecture 4)
def gradient_descent(num_epochs , number_hidden , hid_input_size , act_func , loss_func , weight_init, eta, regpara):
  # Initialise paramaters
  theta = init_network( number_hidden, act_func, weight_init, hid_input_size, net_input_size, net_output_size ) 
  max_accuracy = []

  # Loop
  for i in range(num_epochs):

    grads = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)
    arr  = np.arange(10)
    np.random.shuffle(arr)

    for j in range(0,10):
      ind = arr[j]
      x = x_train[ind,:]
      y = y_train[ind,:]
      h, a, y_hat = forward_propogation(x, act_func, theta, number_hidden)  
      grad_current = backward_propogation(h, a, y, y_hat, theta, number_hidden, hid_input_size, net_input_size, net_output_size, loss_func, act_func, regpara)

      for key in grads:
        grads[key] = grads[key] + grad_current[key]

    for keynew in theta:
      theta[keynew] = theta[keynew] - eta*(grads["d" + keynew]) 
    
    loss_accuracy_compute(i, num_epochs, max_accuracy, x_train, y_train, x_val, y_val, theta, number_hidden, act_func, loss_func, regpara)

  #return theta

# Q3: Implement the backpropagation algorithm with support for the following optimisation functions

# The following functions creates the momenta for weights and biases to be used for 
# updating in the GD algorithm
def init_momenta(number_hidden, hid_input_size, net_input_size, net_output_size):
    size = [] # this contains the list of number of nodes of every layer in the network
    for j in range(number_hidden):
      size.append(hid_input_size)
    size = [net_input_size] + size + [net_output_size]
    momenta = {}
    for i in range(1, number_hidden+2):
        momenta["mW" + str(i)] = np.zeros((size[i], size[i-1]))
        momenta["mb" + str(i)] = np.zeros((size[i],1))
    return momenta

# The following function implements momentum based gradient descent (Lecture 5)
def momentum_based_gradient_descent(num_epochs, number_hidden, hid_input_size, regpara, eta, batch_size, act_func, loss_func, weight_init, net_input_size, net_output_size, gamma ):
    
    # Initialise paramaters
    theta = init_network( number_hidden, act_func, weight_init, hid_input_size, net_input_size, net_output_size )
    prev_momenta = init_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
    momenta = init_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
    points_covered = 0
    max_accuracy = []
    
    # Loop
    for i in range(num_epochs):
      # Initialise gradients
      grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)
      arr  = np.arange(54000)
      np.random.shuffle(arr)

      for j in range(0,54000):
          ind = arr[j]
          x = x_train[ind,:]
          y = y_train[ind,:]
          h, a, y_hat = forward_propogation(x, act_func, theta, number_hidden)   
          grad_t = backward_propogation(h, a, y, y_hat, theta, number_hidden, hid_input_size, net_input_size, net_output_size, loss_func, act_func, regpara)
         
          # weight updates
          for key in grad:
              grad[key] = grad[key] + grad_t[key]

          points_covered = points_covered + 1

          if points_covered % batch_size == 0:
              for keynew in theta:
                momenta["m" + keynew] = gamma*prev_momenta["m"+ keynew] + eta*(grad["d" + keynew])
                theta[keynew] = theta[keynew] - momenta["m" + keynew]
                prev_momenta["m" + keynew] = momenta["m" + keynew] 

              grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)
      
      loss_accuracy_compute(i, num_epochs, max_accuracy, x_train, y_train, x_val, y_val, theta, number_hidden, act_func, loss_func, regpara)

    #return theta

# Nesterov Accelarated Gradient Descent (Lecture 5)

def nesterov_gradient_descent(num_epochs, number_hidden, hid_input_size, regpara, eta, batch_size, act_func, loss_func, weight_init, net_input_size, net_output_size, gamma):
                              
    # Initialise paramaters
    theta = init_network( number_hidden, act_func, weight_init, hid_input_size, net_input_size, net_output_size )
    prev_momenta = init_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
    momenta = init_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
    points_covered = 0
    max_accuracy = []

    # Loop
    for i in range(num_epochs):

        # Initialise gradients
        grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)
        arr  = np.arange(54000)
        np.random.shuffle(arr)

        for j in range(0,54000):
          ind = arr[j]
          x = x_train[ind,:]
          y = y_train[ind,:]
          h, a, y_hat = forward_propogation(x, act_func, theta, number_hidden)   
          grad_t = backward_propogation(h, a, y, y_hat, theta, number_hidden, hid_input_size, net_input_size, net_output_size, loss_func, act_func, regpara)

          # calculating gradients after partial updates 
          for key in grad:
              grad[key] = grad[key] + grad_t[key]
          
          points_covered = points_covered + 1

          if points_covered % batch_size == 0:
              # full update
              for key in theta:
                momenta["m" + key] = gamma*prev_momenta["m"+ key] + eta*(grad["d" + key])
                theta[key] = theta[key] - momenta["m" + key]
                prev_momenta["m" + key] = momenta["m" + key] 
              
              # initialise gradients after every batch is done
              grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)

              # partial updates
              for key in theta:
                momenta["m" + key] = gamma*prev_momenta["m" + key]
                grad["d" + key] = grad["d" + key] - momenta["m" + key]
        

        loss_accuracy_compute(i, num_epochs, max_accuracy, x_train, y_train, x_val, y_val, theta, number_hidden, act_func, loss_func, regpara)
      
    #return theta

# Stochatic gradient descent (Lecture 5)

def stochastic_gradient_descent(num_epochs, number_hidden, hid_input_size, regpara, eta, batch_size, act_func, loss_func, weight_init, net_input_size, net_output_size, gamma):
                                
    # Initialise paramaters
    
    theta = init_network( number_hidden, act_func, weight_init, hid_input_size, net_input_size, net_output_size )
    points_covered = 0
    max_accuracy = []

    # Loop
    for i in range(num_epochs):

        # Initialise gradients
        grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)
        arr  = np.arange(54000)
        np.random.shuffle(arr)

        for j in range(0,54000):
          ind = arr[j]
          x = x_train[ind,:]
          y = y_train[ind,:]
          h, a, y_hat = forward_propogation(x, act_func, theta, number_hidden)   
          grad_t = backward_propogation(h, a, y, y_hat, theta, number_hidden, hid_input_size, net_input_size, net_output_size, loss_func, act_func, regpara)

          for key in grad:
            grad[key] = grad[key] + grad_t[key]
          
          points_covered = points_covered + 1

          if points_covered % batch_size == 0:
            # seen one mini batch
            for key in theta:
              theta[key] = theta[key] - eta*(grad["d" + key])

            # Initialise gradients
            grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)

        loss_accuracy_compute(i, num_epochs, max_accuracy, x_train, y_train, x_val, y_val, theta, number_hidden, act_func, loss_func, regpara)
        
    #return theta  

# The following functions creates the square momenta for to be used in the 
# update rule in the RMSprop, adam optimiser

def init_square_momenta(number_hidden, hid_input_size, net_input_size, net_output_size):
    size = [] # this contains the list of number of nodes of every layer in the network
    for j in range(number_hidden):
      size.append(hid_input_size)
    size = [net_input_size] + size + [net_output_size]
    momenta = {}
    for i in range(1, number_hidden+2):
        momenta["vW" + str(i)] = np.zeros((size[i], size[i-1]))
        momenta["vb" + str(i)] = np.zeros((size[i],1))
    return momenta


# RMSprop (Lecture 5)
def RMSprop(num_epochs, number_hidden, hid_input_size, regpara, eta, batch_size, act_func, loss_func, weight_init, net_input_size, net_output_size, eps, beta):

  # Initialise paramaters

  theta = init_network( number_hidden, act_func, weight_init, hid_input_size, net_input_size, net_output_size )
  prev_momenta = init_square_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
  momenta = init_square_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
  points_covered = 0
  max_accuracy = []

  # Loop
  for i in range(num_epochs):

       # Initialise gradients
      grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)
      arr  = np.arange(54000)
      np.random.shuffle(arr)

      for j in range(0,54000):
        ind = arr[j]
        x = x_train[ind,:]
        y = y_train[ind,:]
        h, a, y_hat = forward_propogation(x, act_func, theta, number_hidden)   
        grad_t = backward_propogation(h, a, y, y_hat, theta, number_hidden, hid_input_size, net_input_size, net_output_size, loss_func, act_func, regpara)

        for key in grad:
            grad[key] = grad[key] + grad_t[key]
          
        points_covered = points_covered + 1

        if points_covered % batch_size == 0:

          # Update rule for RMSprop
          for keynew in theta:
            momenta["v" + keynew] = beta*prev_momenta["v"+ keynew] + (1 - beta)*((grad["d" + keynew])**2)
            theta[keynew] = theta[keynew] - (eta/ (np.sqrt(momenta["v" + keynew] + eps)) )*(grad["d" + keynew]) 
            prev_momenta["v" + keynew] = momenta["v" + keynew] 
          
          # Initialise gradients
          grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)

      loss_accuracy_compute(i, num_epochs, max_accuracy, x_train, y_train, x_val, y_val, theta, number_hidden, act_func, loss_func, regpara)

  #return theta   

# Adam optimiser (Lecture 5)

def adam(num_epochs, number_hidden, hid_input_size, regpara, eta, batch_size, act_func, loss_func, weight_init, net_input_size, net_output_size, eps, beta1, beta2 ):

    # Initialise paramaters

    theta = init_network( number_hidden, act_func, weight_init, hid_input_size, net_input_size, net_output_size )

    prev_momenta = init_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
    momenta = init_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
    momenta_bias = init_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)

    prev_momenta_sq = init_square_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
    momenta_sq = init_square_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
    momenta_sq_bias = init_square_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)

    points_covered = 0
    max_accuracy = []

    # Loop
    for i in range(num_epochs):

      # Initialise gradients
      grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)
      time_count = 0
      arr  = np.arange(54000)
      np.random.shuffle(arr)

      for j in range(0,54000):
          ind = arr[j]
          x = x_train[ind,:]
          y = y_train[ind,:]
          h, a, y_hat = forward_propogation(x, act_func, theta, number_hidden)   
          grad_t = backward_propogation(h, a, y, y_hat, theta, number_hidden, hid_input_size, net_input_size, net_output_size, loss_func, act_func, regpara)

          for key in grad:
              grad[key] = grad[key] + grad_t[key]
            
          points_covered = points_covered + 1

          if points_covered % batch_size == 0:
              time_count = time_count + 1

              for key in theta:

                  # First moments
                  momenta["m" + key] = beta1*prev_momenta["m" + key] + (1 - beta1) * grad["d" + key]
                  momenta_bias["m" + key] = momenta["m" + key]/(1 - np.power(beta1, time_count)) #bias correction

                  # Second moments
                  momenta_sq["v" + key] = beta2*prev_momenta_sq["v" + key] + (1 - beta2)*((grad["d" + key])**2)
                  momenta_sq_bias["v" + key] = momenta_sq["v" + key]/(1 - np.power(beta2, time_count)) #bias correction

                  #w_t+1
                  theta[key] = theta[key] - (eta/np.sqrt(momenta_sq_bias["v" + key] + eps))*momenta_bias["m" + key]

                  prev_momenta["m" + key] = momenta["m" + key]
                  prev_momenta_sq["v" + key] = momenta_sq["v" + key]
              
              # Initialise gradients
              grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)

      loss_accuracy_compute(i, num_epochs, max_accuracy, x_train, y_train, x_val, y_val, theta, number_hidden, act_func, loss_func, regpara)
  
    #return theta

# Nadam optimiser (NAG + Adam)

def nadam(num_epochs, number_hidden, hid_input_size, regpara, eta, batch_size, act_func, loss_func, weight_init, net_input_size, net_output_size, eps, beta1, beta2 ):

    # Initialise paramaters

    theta = init_network( number_hidden, act_func, weight_init, hid_input_size, net_input_size, net_output_size )

    prev_momenta = init_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
    momenta = init_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
    momenta_bias = init_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)

    prev_momenta_sq = init_square_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
    momenta_sq = init_square_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)
    momenta_sq_bias = init_square_momenta(number_hidden, hid_input_size, net_input_size, net_output_size)

    points_covered = 0
    max_accuracy = []

    # Loop
    for i in range(num_epochs):

      # Initialise gradients
      grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)
      time_count = 0
      arr  = np.arange(54000)
      np.random.shuffle(arr)

      for j in range(0,54000):
          ind = arr[j]
          x = x_train[ind,:]
          y = y_train[ind,:]
          h, a, y_hat = forward_propogation(x, act_func, theta, number_hidden)   
          grad_t = backward_propogation(h, a, y, y_hat, theta, number_hidden, hid_input_size, net_input_size, net_output_size, loss_func, act_func, regpara)

          for key in grad:
              grad[key] = grad[key] + grad_t[key]
            
          points_covered = points_covered + 1

          if points_covered % batch_size == 0:
              time_count = time_count + 1

              for key in theta:

                  # First moments
                  momenta["m" + key] = beta1*prev_momenta["m" + key] + (1 - beta1) * grad["d" + key]
                  adam_bias = momenta["m" + key]/(1 - np.power(beta1, time_count)) 
                  momenta_bias["m" + key] = beta1*adam_bias + (1 - beta1) * grad["d" + key] #bias correction

                  # Second moments
                  momenta_sq["v" + key] = beta2*prev_momenta_sq["v" + key] + (1 - beta2)*((grad["d" + key])**2)
                  momenta_sq_bias["v" + key] = momenta_sq["v" + key]/(1 - np.power(beta2, time_count)) #bias correction

                  #w_t+1
                  theta[key] = theta[key] - (eta/np.sqrt(momenta_sq_bias["v" + key] + eps))*momenta_bias["m" + key]

                  prev_momenta["m" + key] = momenta["m" + key]
                  prev_momenta_sq["v" + key] = momenta_sq["v" + key]
              
              # Initialise gradients
              grad = init_grad(number_hidden, hid_input_size, net_input_size, net_output_size)

      #val_acc, val_loss, train_acc, train_loss = loss_accuracy_compute(i, num_epochs, max_accuracy, x_train, y_train, x_val, y_val, theta, number_hidden, act_func, loss_func, regpara)
      loss_accuracy_compute(i, num_epochs, max_accuracy, x_train, y_train, x_val, y_val, theta, number_hidden, act_func, loss_func, regpara)
  
    #return theta, val_acc, val_loss, train_acc, train_loss

# Configure the sweep 
# Specify the method, metric, parameters to search through

sweep_config = {
    'method': 'bayes',
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
    'early_terminate': {
            'type': 'hyperband',
            'min_iter': [3],
            's': [2]
    },
    'parameters': {
        'epochs': {
            'values': [5, 10] #number of epochs
        },
        'number_hidden': {
            'values': [3, 4, 5] #number of hidden layers
        },
        'hidden_inputsize': {
            'values': [32, 64, 128] #size of every hidden layer
        },
        'weight_decay': {
            'values': [0, 0.0005,  0.5] #L2 regularisation
        },
        'learning_rate': {
            'values': [1e-3, 1e-4] 
        },
        'optimizer': {
            'values': [ 'sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
        },
        'batch_size' : {
            'values':[16, 32, 64]
        },
        'weight_init': {
            'values': ['random','xavier']
        },
        'activation': {
            'values': ['sigmoid','tanh','relu']
        }
        
        }
}

# Initialize a new sweep
# Arguments:
# param_config: the sweep config dictionary defined above
# entity: Set the username for the sweep
# project: Set the project name for the sweep
sweep_id = wandb.sweep(sweep_config, entity="bharatik", project="cs6910assignment1")

# The sweep calls this function with each set of hyperparameters

def train():
    # Default values for hyper-parameters that we're going to sweep over
    config_defaults = {
        'epochs': 5,
        'number_hidden': 3,
        'hidden_inputsize': 32,
        'weight_decay': 0,
        'learning_rate': 1e-3,
        'optimizer': 'momentum',
        'batch_size': 64,
        'activation': 'sigmoid',
        'weight_init': 'random',
        'loss' : 'squared',
        'gamma' : 0.9, # update parameter
        'net_input_size' : 784, 
        'net_output_size' : 10,
        'eps' : 1e-8,
        'beta': 0.95,
        'beta1' : 0.9,
        'beta2' : 0.999
    }

     # Initializing a new wandb run
    #wandb.init(config=config_defaults,resume=True)
    
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config 
    wandb.run.name = "hl_" + str(config.hidden_inputsize)+"_bs_"+str(config.batch_size)+"_ac_"+ config.activation + "_loss_" + config.loss

    # Defining the various optimizers

    if config.optimizer=='adam':
      adam(num_epochs = config.epochs, number_hidden = config.number_hidden, hid_input_size = config.hidden_inputsize, regpara = config.weight_decay, eta = config.learning_rate, batch_size = config.batch_size, act_func = config.activation, loss_func = config.loss, weight_init = config.weight_init, net_input_size = config.net_input_size, net_output_size = config.net_output_size, eps = config.eps, beta1=config.beta1, beta2 = config.beta2)
    elif config.optimizer=='nadam':
      nadam(num_epochs = config.epochs, number_hidden = config.number_hidden, hid_input_size = config.hidden_inputsize, regpara = config.weight_decay, eta = config.learning_rate, batch_size = config.batch_size, act_func = config.activation, loss_func = config.loss, weight_init = config.weight_init, net_input_size = config.net_input_size, net_output_size = config.net_output_size, eps = config.eps, beta1=config.beta1, beta2 = config.beta2)
    elif config.optimizer=='sgd':
      stochastic_gradient_descent(num_epochs = config.epochs, number_hidden = config.number_hidden, hid_input_size = config.hidden_inputsize, regpara = config.weight_decay, eta = config.learning_rate, batch_size = config.batch_size, act_func = config.activation, loss_func = config.loss, weight_init = config.weight_init, net_input_size = config.net_input_size, net_output_size = config.net_output_size, gamma = config.gamma)
    elif config.optimizer=='rmsprop':
      RMSprop(num_epochs = config.epochs, number_hidden = config.number_hidden, hid_input_size = config.hidden_inputsize, regpara = config.weight_decay, eta = config.learning_rate, batch_size = config.batch_size, act_func = config.activation, loss_func = config.loss, weight_init = config.weight_init, net_input_size = config.net_input_size, net_output_size = config.net_output_size, eps = config.eps, beta=config.beta)
    elif config.optimizer=='momentum':
      momentum_based_gradient_descent(num_epochs = config.epochs, number_hidden = config.number_hidden, hid_input_size = config.hidden_inputsize, regpara = config.weight_decay, eta = config.learning_rate, batch_size = config.batch_size, act_func = config.activation, loss_func = config.loss, weight_init = config.weight_init, net_input_size = config.net_input_size, net_output_size = config.net_output_size, gamma = config.gamma)
    elif config.optimizer=='nesterov':
      nesterov_gradient_descent(num_epochs = config.epochs, number_hidden = config.number_hidden, hid_input_size = config.hidden_inputsize, regpara = config.weight_decay, eta = config.learning_rate, batch_size = config.batch_size, act_func = config.activation, loss_func = config.loss, weight_init = config.weight_init, net_input_size = config.net_input_size, net_output_size = config.net_output_size, gamma = config.gamma)

wandb.agent('91idz382',train,count=1)

#Best hyperparameters for validation accuracy of 0.88717

best_num_epochs = 10
best_number_hidden = 3
best_hid_input_size = 128
best_regpara = 0.0005
best_eta = 0.001
best_batch_size = 16
best_act_func = 'relu'
best_loss_func = 'cross'
best_weight_init = 'xavier'

theta  = nadam(num_epochs = best_num_epochs, number_hidden = best_number_hidden, hid_input_size = best_hid_input_size, regpara = best_regpara, eta = best_eta, batch_size = best_batch_size, act_func = best_act_func, loss_func = best_loss_func, weight_init = best_weight_init, net_input_size = 784, net_output_size = 10, eps = 1e-8, beta1 = 0.9, beta2=0.999)

def test_loss_accuracy_compute(best_num_epochs, x_test, y_test, theta, best_number_hidden, best_act_func, best_loss_func, best_regpara):

    sum_norm = 0
    for i in range(1,best_number_hidden + 2):
        sum_norm = sum_norm + np.sum(np.square(theta["W"+str(i)]))

    test_loss = 0 # training loss
    ctr = 0
    true_label = []
    pred_label = []
    for j in range(0,10000):
        x = x_test[j,:]
        y = y_test[j,:]
        h,a,y_hat = forward_propogation(x, best_act_func, theta, best_number_hidden)
        
      
        if best_loss_func == "cross":
            test_loss = test_loss + (cross_entropy_loss(y_hat,y,sum_norm, best_regpara))
        else:
            test_loss = test_loss + (squared_loss(y, y_hat,sum_norm, best_regpara))
           
        # convert one hot encoded vector to label 
        y = np.argmax(y, axis = 0)
        y_hat = np.argmax(y_hat, axis = 0)
        true_label.append(y)
        pred_label.append(y_hat)

        if y == y_hat:
          ctr = ctr + 1


    test_acc = ctr/10000.0
    test_loss = test_loss/10000.0

    return true_label, pred_label, test_acc, test_loss


true_label, pred_label, test_acc, test_loss = test_loss_accuracy_compute(best_num_epochs, x_test, y_test, theta, best_number_hidden, best_act_func, best_loss_func, best_regpara)
ctr = 0

for i in range(10000): 
  if true_label[i] == pred_label[i].tolist():
    ctr = ctr + 1

test_accuracy = ctr/10000.0

#Plotting confusion matrix

cm = confusion_matrix(true_label, pred_label)

#wandb.init(project="cs6910assignment1", entity="bharatik")
img  = pl.matshow(cm)
pl.title('Confusion matrix of the classifier')
pl.colorbar()
pl.show()

#wandb.log({"confusion matrix": [ wandb.Image(img, caption='confusion matrix') ]})

#MNIST handwritten

#wandb.init(project="cs6910assignment1", entity="bharatik")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Split data for cross validation
x_val = x_train[54000:]
y_val = y_train[54000:]
x_train = x_train[:54000]
y_train = y_train[:54000]  

image = [];
label = [];
for i in range(54000):
  if len(label) >= 10:
    break;
  if class_names[y_train[i]] not in label:
      image.append(x_train[i])
      label.append(class_names[y_train[i]])
#wandb.log({"Examples": [ wandb.Image(img, caption=caption) for img, caption in zip(image,label)]})

# Vectorise and normalize the data
x_train = x_train.reshape(x_train.shape[0], 784)
x_val  = x_val.reshape(x_val.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

x_train = x_train / 255.0
x_test = x_test / 255.0
x_val  = x_val / 255.0

# One hot encoding for labels
y_train = to_categorical(y_train)
y_val   = to_categorical(y_val)
y_test = to_categorical(y_test)

#theta, val_acc, val_loss, train_acc, train_loss = nadam(num_epochs = best_num_epochs, number_hidden = best_number_hidden, hid_input_size = best_hid_input_size, regpara = best_regpara, eta = best_eta, batch_size = best_batch_size, act_func = best_act_func, loss_func = best_loss_func, weight_init = best_weight_init, net_input_size = 784, net_output_size = 10, eps = 1e-8, beta1 = 0.9, beta2=0.999)
#true_label, pred_label, test_acc, test_loss = test_loss_accuracy_compute(best_num_epochs, x_test, y_test, theta, best_number_hidden, best_act_func, best_loss_func, best_regpara)
ctr = 0

for i in range(10000): 
  if true_label[i] == pred_label[i].tolist():
    ctr = ctr + 1

test_accuracy = ctr/10000.0