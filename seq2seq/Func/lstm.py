
import tensorflow as tf

class lstm:
    def __init__(self,input_size, num_nodes):
        self._num_nodes = num_nodes
        self.xx = tf.Variable(tf.truncated_normal([input_size, self._num_nodes * 4], -0.1, 0.1))
        self.mm = tf.Variable(tf.truncated_normal([self._num_nodes, self._num_nodes * 4], -0.1, 0.1))
        self.bb = tf.Variable(tf.zeros([1, self._num_nodes * 4]))


                          
    def lstm_cell(self,input,output,state):
        #global dropout
        #i=tf.nn.dropout(i,keep_prob=dropout)
        matmuls = tf.matmul(input, self.xx)+ tf.matmul(output, self.mm) + self.bb        
        input_gate  = tf.sigmoid(matmuls[:, 0 * self._num_nodes : 1 * self._num_nodes])
        forget_gate = tf.sigmoid(matmuls[:, 1 * self._num_nodes : 2 * self._num_nodes])
        update      =            matmuls[:, 2 * self._num_nodes : 3 * self._num_nodes]
        output_gate = tf.sigmoid(matmuls[:, 3 * self._num_nodes : 4 * self._num_nodes])
        state       = forget_gate * state + input_gate * tf.tanh(update)
        output=output_gate * tf.tanh(state)
        return output, state