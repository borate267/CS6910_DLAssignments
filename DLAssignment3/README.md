# Sequence to Sequence model (transliteration)
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
                batch_size = 128, beam_width = 5, num_enc = 2,num_dec = 3)
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
The function **build_fit** finds the accuracy of the model with test data for both vanilla and attention models.

- #### Attention visualization
The function **attention_plot** plots the attention heatmaps for multiple inputs.

```python
model_rnn.attention_plot(val_encoder_input_data)
```
## Run

In a terminal or command window, navigate to the top-level project directory `CS6910_DLAssignments/DLAssignment3/` (that contains this README) and run the following command for wandb sweep:

```bash
jupyter notebook transliteration_keras.ipynb
```

```bash
jupyter notebook transliteration_attention.ipynb
``` 

The code for evaluating the perfomance of the Seq2Seq model using Dakshina dataset can be run using the following command:
```bash
jupyter notebook Testing_transliteration_keras.ipynb
``` 
The code for evaluating the perfomance of the Seq2Seq model with attention using Dakshina dataset can be run using the following command:
```bash
jupyter notebook Testing_attention.ipynb
``` 
The code for visualizing the attention vectors of the Seq2Seq model using Dakshina dataset can be run using the following command:
```bash
jupyter notebook visualize_attention.ipynb
``` 
The code for visualizing memorization in RNNs of the Seq2Seq model using Dakshina dataset can be run using the following command:
```bash
jupyter notebook final_q6.ipynb
``` 

## Data
The Daskshina dataset is uploaded in this git repository and imported using the "git clone" command:
```python
!git clone https://github.com/borate267/lexicon-dataset.git
```

### Data Preprocessing
- The transliteration task is carried for Tamil lexicons from Google's Dakshina dataset.
- The Tamil lexicon contains 68218 training words, 6827 validation words and 6864 testing words.
- Tokenization of the characters in the dataset are performed using numpy.

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
