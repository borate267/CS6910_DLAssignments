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

- #### Attention
The function **model_test** finds the accuracy of the model with test data.

- #### Attention visualization
The 

```python
['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception']
```
- #### Attention memory visualization
The

## Run

In a terminal or command window, navigate to the top-level project directory `CNN_Pytorch/` (that contains this README) and run one of the following commands:

```bash
ipython notebook transliteration_keras.ipynb
```  
or
```bash
jupyter notebook transliteration_keras.ipynb
```
The code for evaluating the perfomance of the custom CNN model with iNaturalist dataset is seperately uploaded and it can be run using the following command:
```bash
jupyter notebook Testing_transliteration_keras.ipynb
``` 
The code for guided backpropagation of the custom CNN models with iNaturalist dataset is seperately uploaded and it can be run using the following command:
```bash
jupyter notebook transliteration_attention.ipynb
``` 
The code for evaluating the perfomance of the pretrained CNN models with iNaturalist dataset is seperately uploaded and it can be run using the following command:
```bash
jupyter notebook Testing_attention.ipynb
``` 

```bash
jupyter notebook visualize_attention.ipynb
``` 

## Data
The Daskshina dataset is uploaded in this git repository and imported using the "git clone" command:
```python
!git clone https://github.com/borate267/lexicon-dataset.git
```

### Data Preprocessing
- The iNaturalist dataset contains 9999 training image data and 2000 testing.
- The training data is very split with ratio of 90:10 for training and validation. This is done to avoid overfitting.
- All the test, train and validation data are imported using the data loader function in torch library.
- The transfromers function are used for resizing, cropping, normalizng the images and then convert it to tensors.

## Report link
[assignment_3_report](https://wandb.ai/cs6910assignment3/RNN/reports/FDL-Assignment-3---Vmlldzo2NzE5MTM)

## Reference
- https://keras.io/examples/nlp/lstm_seq2seq/#prepare-the-data
- https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
- https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/
- https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/
- https://towardsdatascience.com/visualising-lstm-activations-in-keras-b50206da96ff
- https://medium.com/datalogue/attention-in-keras-1892773a4f22
- https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/#:~:text=2.-,Keras%20Embedding%20Layer,API%20also%20provided%20with%20Keras.
