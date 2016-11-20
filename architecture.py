import tensorflow as tf
import numpy as np
import sys
from tensorflow_ops import _conv_layer, _fc_layer, lrelu


'''
   Makes a prediction for training
'''
def predict(states, actions):
   conv1 = lrelu(_conv_layer(states, 8, 4, 16, 'p_conv1'))
   conv2 = lrelu(_conv_layer(conv1, 4, 2, 32, 'p_conv2'))

   fc1 = lrelu(_fc_layer(conv2, 256, True, 'p_fc1'))
   fc2 = lrelu(_fc_layer(fc1, 6, False, 'p_fc2'))

   return fc2


'''
   Trains the network. This will return values for all possible actions.
   When those values get returned, take the argmax
'''
def train(states):
   conv1 = lrelu(_conv_layer(states, 8, 4, 16, 't_conv1'))
   conv2 = lrelu(_conv_layer(conv1, 4, 2, 32, 't_conv2'))

   fc1 = lrelu(_fc_layer(conv2, 256, True, 't_fc1'))
   fc2 = lrelu(_fc_layer(fc1, 6, False, 't_fc2'))
   return fc2


'''
   Makes a prediction that is used in the actual emulator
'''
def inference(states):
   conv1 = lrelu(_conv_layer(states, 8, 4, 16, 'i_conv1'))
   conv2 = lrelu(_conv_layer(conv1, 4, 2, 32, 'i_conv2'))

   fc1 = lrelu(_fc_layer(conv2, 256, True, 'i_fc1'))
   fc2 = _fc_layer(fc1, 6, False, 'i_fc2')
   
   # this returns 6 actions
   return fc2
 
def loss(predicted_value, actual_value, gamma):
   error = tf.reduce_sum(predicted_value - (gamma*actual_value))
   return error 
