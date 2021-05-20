# Recurrent Neural Networks (RNN)
# Fundamentals of DL course Assignment - 3 


## Install

This project requires **Python** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [TensorFlow](https://www.tensorflow.org/install)
- [wandb](https://wandb.ai/site)
- [wget](https://pypi.org/project/wget/)

## Code
- #### Recurrent Neural network Class (MyRNN)
The recurrent neural network based encoder-decoder (seq2seq) is built using tensorflow Keras. The function **build_fit** contains the encoder-decoder pipeline with input embedding (input embedding, encoder layers, decoding layers, back_prop, update_parameters) is used for computing the training and the validation error along with validation accuracy. 

The following line of code is an example to define a model using the MyNN class:

```python
model_rnn = MyRNN(cell_type = 'LSTM', in_emb = 128, hidden_size=128,
                learning_rate= 0.01, dropout=0.2,pred_type = 'beam_search',epochs = 10,
                batch_size = 128, beam_width = 10, num_enc = 2,num_dec = 3)

```
After defining the model, the training of the model can be done using the following command:
```python
  model_rnn.build_fit(encoder_input_data,decoder_input_data,decoder_target_data,x_test, y_test)
```
- #### Wandb configuration
Wandb is a tool for tuning the hyper-parameters of a model. The wandb sweep requires to define a sweep configuaration with hyper-parameters in a dictionary type. The following code snippet is an example of defining the wandb sweep configuration:
```python
sweep_config = {
    'method': 'bayes', 
    'metric': {
      'name': 'val_accuracy',
      'goal': 'maximize'   
    },
    'parameters': {

        'dropout': {
            'values': [0.0, 0.1, 0.2]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'batch_size': {
            'values': [64, 128]
        },
        'in_emb': {
            'values': [32, 64, 128]
        },
        'num_enc': {
            'values': [1, 2, 3]
        },
        'num_dec': {
            'values': [1, 2, 3]
        },
        'hidden_size':{
            'values': [32, 64, 128]
        },
        'cell_type': {
            'values': ['RNN', 'GRU', 'LSTM']
        },
        'dec_search': {
            'values': ['beam_search', 'greedy']
        },
        'beam_width':{
            'values': [3,5]
        }
    }
}
```
```python
wandb.agent("auh90ups", entity="cs6910assignment3",project="RNN", function =train_sweep,count=100)
```
- #### Train sweep function
The function **train_sweep** is the main function called by the wandb sweep. This function contains the sweep configurations, and the seq2seq model.  

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
The iNaturalist dataset is downloaded directly from the downloadable link using the following the "wget" command:
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
