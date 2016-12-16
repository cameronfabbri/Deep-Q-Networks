import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../ops/')
from tf_ops import conv2d, fc_layer, lrelu



# fc_layer(inputs, hidden_units, flatten, name):
# conv2d(inputs, kernel_size, stride, num_features, name):

'''
   Makes a prediction for training
'''
def predict(states, actions):
   conv1 = lrelu(conv2d(states, 8, 4, 16, 'p_conv1'))
   conv2 = lrelu(conv2d(conv1, 4, 2, 32, 'p_conv2'))

   fc1 = lrelu(fc_layer(conv2, 256, True, 'p_fc1'))
   fc2 = lrelu(fc_layer(fc1, 6, False, 'p_fc2'))

   return fc2


'''
   Trains the network. This will return values for all possible actions.
   When those values get returned, take the argmax
'''
def train_network(states):
   conv1 = lrelu(conv2d(states, 8, 4, 16, 't_conv1'))
   conv2 = lrelu(conv2d(conv1, 4, 2, 32, 't_conv2'))

   fc1 = lrelu(fc_layer(conv2, 256, True, 't_fc1'))
   fc2 = lrelu(fc_layer(fc1, 6, False, 't_fc2'))
   return fc2


'''
   Makes a prediction that is used in the actual emulator
'''
def inference(states):
   conv1 = lrelu(conv2d(states, 8, 4, 16, 'i_conv1'))
   conv2 = lrelu(conv2d(conv1, 4, 2, 32, 'i_conv2'))

   fc1 = lrelu(fc_layer(conv2, 256, True, 'i_fc1'))
   fc2 = fc_layer(fc1, 6, False, 'i_fc2')
   
   # this returns 6 actions
   return fc2
 
def loss(predicted_value, actual_value, gamma):
   error = tf.reduce_sum(predicted_value - (gamma*actual_value))
   return error 
