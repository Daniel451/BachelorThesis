import tensorflow as tf
from tensorflow.models.rnn.rnn_cell import BasicRNNCell
from tensorflow.python.ops.rnn_cell import linear as tflinear




class CustomRNNCell(BasicRNNCell):

    def __init__(self, num_units, activation_function=tf.nn.tanh):
        super(CustomRNNCell, self).__init__(num_units)
        self.act_func = activation_function
    
    def __call__(self, inputs, state, scope=None):
        """Most basic RNN: output = new_state = tanh(W * input + U * state + B)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicRNNCell"
            output = tflinear([inputs, state], self._num_units, True)
            output = self.act_func(output)
            return output, output
