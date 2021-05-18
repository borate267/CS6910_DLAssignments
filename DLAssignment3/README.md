# Recurrent Neural Networks (CNN)
# Fundamentals of DL course Assignment - 3 


## Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [Pytorch](https://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/install)
- [OpenCV](https://pypi.org/project/opencv-python/)
- [wandb](https://wandb.ai/site)
- [wget](https://pypi.org/project/wget/)
- [mapextrackt](https://pypi.org/project/mapextrackt/)

## Code
- #### Convolution Neural network Class (MyNN)
The convolution neural network is built using Pytorch with conditions on batch normalization. The function **fit** is the contains the neural network pipeline (Forward_prog, compute_loss, back_prop, update_parameters) is used for computing the training and the validation error along with validation accuracy. 

The **train** function consists of data-preprocessing, initializing the model and it's parameters.

The following line of code is an example to define a model using the MyNN class:

```python
model = MyNN(drop_out = 0.2, batch_norm = 'Yes', filter_n = 128,  filter_org = 'double_down', hidden_out =196)
```
After defining the model, the training of the model can be done using the following command:
```python
fit(model,train_loader,val_loader,optimizer,epoch,criterion)
```
- #### Wandb configuration
Wandb is a tool for tuning the hyper-parameters of a model. The wandb sweep requires to define a sweep configuaration with hyper-parameters in a dictionary type. The following code snippet is an example of defining the wandb sweep configuration:
```python
sweep_config = {
    'method': 'bayes', #grid, random
    'metric': {
      'name': 'accuracy',
      'goal': 'maximize'   
    },
    'parameters': {
        'drop_out': {
            'values': [0.2, 0.3]
        },
        'batch_norm': {
            'values': ['Yes', 'No']
        },
        'filter_n': {
            'values': [64, 128]
        
        },
        'batch_size': {
            'values': [16, 32]
        },
        'filter_org': {
            'values': ['same', 'double_up', 'double_down']
        },
        'epoch': {
            'values': [5,10]
        },
        'data_aug': {
            'values': ['Yes', 'No']
        },
        'optimizer': {
            'values': ['SGD','ADAM'] 
        },
        'lr': {
            'values': [0.1,0.01] 
        },
        'hidden_out': {
            'values': [128,196] 
        },
    }
}
```

```python
sweep_id = wandb.sweep(sweep_config, entity="paddy3696", project="cnn_inat")
```
- #### Train sweep function
The function **train** is the main function called by the wandb sweep. This function contains the wandb initialization and data pre-processing.  

- #### Testing
The function **model_test** finds the accuracy of the model with test data.

- #### Filter visualization
The filter visualization is performed using **mapextrackt** library. feature extraction and visualization functions is used from this library. 

- #### Pre - trained models
The pre-trained models are trained used from the pytorch library. These models are initialized as a hyper-parameter. The list of pre-trained models is mentioned below:

```python
['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
```
- #### Guided Backpropogation
The Guided backpropogation is performed using Tensorflow Keras because extracting features of each layer was not direct. Another significant reason for using Keras instead of Pytorch is to understand the difference between two libraries. The same CNN model is built completely from the scratch and fifth convolution layer is visualized using the guided propogat

## Run

In a terminal or command window, navigate to the top-level project directory `CNN_Pytorch/` (that contains this README) and run one of the following commands:

```bash
ipython notebook Inat_cnn_train.ipynb
```  
or
```bash
jupyter notebook Inat_cnn_train.ipynb
```
The code for evaluating the perfomance of the custom CNN model with iNaturalist dataset is seperately uploaded and it can be run using the following command:
```bash
jupyter notebook Inat_cnn_test.ipynb
``` 
The code for guided backpropagation of the custom CNN models with iNaturalist dataset is seperately uploaded and it can be run using the following command:
```bash
jupyter notebook guided_backprop.ipynb
``` 
The code for evaluating the perfomance of the pretrained CNN models with iNaturalist dataset is seperately uploaded and it can be run using the following command:
```bash
jupyter notebook inat_cnn_pretrained.ipynb
``` 

## Data
The iNaturalist datasetis downloaded directly from the downloadable link using the following the "wget" command:
```python
wget.download('https://storage.googleapis.com/wandb_datasets/nature_12K.zip')
```

### Data Preprocessing
- The iNaturalist dataset contains 9999 training image data and 2000 testing.
- The training data is very split with ratio of 90:10 for training and validation. This is done to avoid overfitting.
- All the test, train and validation data are imported using the data loader function in torch library.
- The transfromers function are used for resizing, cropping, normalizng the images and then convert it to tensors.

## Report link
[assignment2_report](https://wandb.ai/paddy3696/cnn_inat/reports/FDL-Assignment-2---Vmlldzo2MDg3Mzg?accessToken=l08ezysoh00yvd68sdpq7r78rvq5l2zjaxbjg6li81d982eu2we6xqky99wuol3r)

## Reference
- Udacity Deep Learning course
- [Sentdex Pytorch tutorials](https://youtube.com/playlist?list=PLQVvvaa0QuDdeMyHEYc0gxFpYwHY2Qfdh)
- [Python Engineer Pytorch tutorials](https://youtube.com/playlist?list=PLqnslRFeH2UrcDBWF5mfPGpqQDSta6VK4)
- https://towardsdatascience.com/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e
- [mapextrackt](https://pypi.org/project/mapextrackt/)
- https://stackoverflow.com/questions/55924331/how-to-apply-guided-backprop-in-tensorflow-2-0
- https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
- [YoloV3](https://pjreddie.com/darknet/yolo/)
- https://www.pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/
