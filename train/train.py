from collections import deque
from tqdm import tqdm
import random
import gym
import sys
import numpy as np

import tensorflow as tf
sys.path.insert(0, '../ops/')
sys.path.insert(0, '../architecture/')

from data_ops import preprocess

from architecture import train_network

def select_action(s_t, epsilon=0.1):
   if np.random.rand() < epsilon:
      a = np.random.randint(6)
   else:
      print 'getting highest ranked action from network...'
      a = np.random.randint(6)

   return a


def train(env, batch_size):
   with tf.Graph().as_default():
      global_step = tf.Variable(0, name='global_step', trainable=False)

      s_t_placeholder  = tf.placeholder(tf.float32, shape=(batch_size, 84, 84, 1))
      #s_t1_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 84, 84, 1))

      action_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 6))

      # reward from the database
      # y_j_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 1))
      #target_reward = train_network(s_t1_placeholder)
      target_reward_placeholder = tf.placeholder(tf.float32, shape=(batch_size,6))

      # actual_y is the result of sending s_j and a_j from the database to the network
      actual_reward = train_network(s_t_placeholder, action=action_placeholder)

      # placeholders for the values coming from the networks to the loss function
      #t_r_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 1))
      #a_r_placeholder = tf.placeholder(tf.float32, shape=(batch_size, 1))

      loss = tf.nn.l2_loss(target_reward_placeholder-actual_reward)
      #loss = tf.nn.l2_loss(t_r_placeholder-a_r_placeholder)

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

         env.render()
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
         s_j = np.asarray(s_j)
         s_j = np.expand_dims(s_j, 0)
         s_j = np.expand_dims(s_j, 3)
         s_j1 = np.asarray(s_j1)
         s_j1 = np.expand_dims(s_j1, 0)
         s_j1 = np.expand_dims(s_j1, 3)

         # send s_j through network, use highest action and predict reward
         # minimize the reward you got with the actual reward
         # y_j is the target reward!
         if terminal_j: # no more steps so no more future reward from sample
            y_j = r_j
         else:
            # send s_j1 to the network and take the highest reward -> getting future expected reward
            #target_reward = np.max(sess.run([target_reward_placeholder], feed_dict={s_t1_placeholder:s_j1})[0][0])
            ones_action = [1,1,1,1,1,1]
            ones_action = np.expand_dims(ones_action,0)
            target_reward = np.max(sess.run([actual_reward], feed_dict={s_t_placeholder:s_j1, action_placeholder:ones_action}))
            y_j = r_j + gamma*target_reward

         target_reward_vec = np.zeros(6)
         target_reward_vec[a_j] = target_reward
         target_reward_vec = np.expand_dims(target_reward_vec, 0)

         action = np.zeros(6)
         action[a_j] = 1
         action = np.expand_dims(action,0)

         actual_y = sess.run([actual_reward], feed_dict={s_t_placeholder:s_j, action_placeholder:action})[0][0]

         _, loss_ = sess.run([train_op, loss], feed_dict={target_reward_placeholder:target_reward_vec, action_placeholder:action, s_t_placeholder:s_j})

         print 'loss:',loss_


if __name__ == '__main__':
   env = gym.make('Breakout-v0')

   batch_size = 1

   train(env, batch_size)


