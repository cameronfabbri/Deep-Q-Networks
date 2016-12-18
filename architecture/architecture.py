import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../ops/')
from tf_ops import conv2d, fc_layer, lrelu



'''
   Trains the network. This will return values for all possible actions.
   When those values get returned, take the argmax.

   The number of actions determines the output layer size
'''
def q_network(states, action_mask, num_actions, reuse=False):

   if reuse:
      tf.get_variable_scope().reuse_variables()

   conv1 = lrelu(conv2d(states, 8, 4, 16, 't_conv1'))
   conv2 = lrelu(conv2d(conv1, 4, 2, 32, 't_conv2'))

   fc1 = lrelu(fc_layer(conv2, 256, True, 't_fc1'))
   fc2 = lrelu(fc_layer(fc1, num_actions, False, 't_fc2'))

   # Multiply the weights by an action mask.
   return tf.mul(fc2, action_mask)



def loss(predicted_value, actual_value, gamma):
   error = tf.reduce_sum(predicted_value - (gamma*actual_value))
   return error
