from collections import deque
from tqdm import tqdm
import universe
import random
import gym
import sys
import numpy as np

import tensorflow as tf
sys.path.insert(0, '../ops/')
sys.path.insert(0, '../architecture/')

from data_ops import preprocess
from tf_ops import select_action

from architecture import train_network

# maybe at some point if i'm taking things out of the experience replay,
# take out the things with lowest rewards

def train(env, batch_size):
   with tf.Graph().as_default():
      global_step = tf.Variable(0, name='global_step', trainable=False)

      state_t   = tf.placeholder(tf.float32, shape=(batch_size, 84, 84, 1))
      state_t_1 = tf.placeholder(tf.float32, shape=(batch_size, 84, 84, 4))
      action    = tf.placeholder(tf.float32, shape=(batch_size, 6))

      target_y = train_network(state_t)

      actual_y = tf.placeholder(tf.float32, shape=(batch_size))

      loss = tf.reduce_mean(tf.square(target_y - actual_y))

      train_op = tf.train.RMSPropOptimizer(learning_rate=1e-5, momentum=0.95, epsilon=0.01).minimize(loss)

      variables = tf.all_variables()
      init      = tf.initialize_all_variables()
      sess      = tf.Session()
      sess.run(init)

      epsilon = 1
      gamma   = 0.99

      experience_replay = []

      x_t = env.reset() # reset the environment
      s_t = preprocess(x_t)

      while True:

         # choose action
         a_t = select_action(s_t, epsilon=epsilon) # select action using e-greedy policy

         # execute action
         x_t1, r_t, terminal, info = env.step(a_t) # take a step and get reward from action
         s_t1 = preprocess(x_t1) # preprocess image

         # store
         e = [s_t, a_t, r_t, s_t1, terminal]
         experience_replay.append(e)

         # sample randomly from experience replay ---- this is what you train on!
         s_j, a_j, r_j, s_j1, terminal_j = random.choice(experience_replay)
         s_j1 = np.asarray(s_j1)
         s_j1 = np.expand_dims(s_j1, 0)
         s_j1 = np.expand_dims(s_j1, 3)

         # send s_j through network, use highest action and predict reward
         # minimize the reward you got with the actual reward

         # y_j is the target reward!
         if terminal_j: # no more steps so no more future reward from sample
            y_j = r_j
         else:
            target_reward = sess.run([target_y], feed_dict={state_t:s_j1})[0][0][a_j]
            y_j = r_j + gamma*target_reward

         
         _, loss_ = sess.run([train_op, loss], feed_dict={target_y:y_j, actual_y:y_})

         print 'loss:',loss_



if __name__ == '__main__':
   env = gym.make('Breakout-v0')

   batch_size = 1

   train(env, batch_size)


