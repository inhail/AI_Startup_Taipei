
import tensorflow as tf
from .lstm import lstm


class seq2seq():
    def __init__(self,vocabulary_size, num_nodes):
        self._vocabulary_size = vocabulary_size
        self._num_nodes = num_nodes
        self._encoder_lstm=lstm(self._vocabulary_size, self._num_nodes)
        self._decoder_lstm=lstm(self._vocabulary_size, self._num_nodes)

    def encoder(self, train_inputs, output, state):
        '''   
        train_input : array of size num_unrolling, each array element is a Tensor of dimension batch_size,
        vocabulary size.
        
        Returns:
        output : Output of LSTM aka Hidden State
        state : Cell state of the LSTM
        
        '''
        i = len(train_inputs) - 1
        while i >= 0:
            en_output, en_state = self._encoder_lstm.lstm_cell(train_inputs[i], output, state)
            i=i-1
        #Return the last output of the lstm cell for decoding
        return en_output ,en_state

    def training_decoder(self, decoder_input, output, state):
        final_outputs=[]
        #Predict the first character using the EOS Tag. We use EOS tag as the start tag
        de_output, de_state = self._decoder_lstm.lstm_cell(decoder_input[-1], output, state)
        final_outputs.append(de_output)
        #Now predict the next outputs using the training labels itself. Using y(n-1) to predict y(n)
        for i in decoder_input[0:-1]:
            de_output,de_state=self._decoder_lstm.lstm_cell(i, de_output, de_state)
            final_outputs.append(de_output)
        return final_outputs, de_output, de_state
    
    
    def inference_decoder(self, go_char, decode_steps, output, state, weight, bias):
        final_outputs=[]
        #First input to decoder is the the Go Character
        de_output, de_state=self._decoder_lstm.lstm_cell(go_char, output, state)
        final_outputs.append(de_output)
        for i in range(decode_steps-1):
            #Feed the previous output as the next decoder input
            decoder_input=tf.nn.softmax(tf.nn.xw_plus_b(de_output, weight, bias))
            de_output,de_state=self._decoder_lstm.lstm_cell(decoder_input, de_output, de_state)
            final_outputs.append(de_output)
        return final_outputs, de_output, de_state