from collections import deque
import time
from tqdm import tqdm
import random
import gym
import sys
import numpy as np

import tensorflow as tf
sys.path.insert(0, '../ops/')
sys.path.insert(0, '../architecture/')

from data_ops import preprocess
from tf_ops import linear_annealing
from architecture import q_network


def train(env, batch_size, num_actions, experience_replay, history_length):
   with tf.Graph().as_default():

      global_step         = tf.Variable(0, name='global_step', trainable=False)
      s_t_placeholder     = tf.placeholder(tf.float32, shape=(batch_size, 84, 84, 4))

      # this is the current frames we're sending to the network to predict an action
      c_frame_placeholder = tf.placeholder(tf.float32, shape=(1, 84, 84, history_length))
      action_placeholder  = tf.placeholder(tf.float32, shape=(batch_size, num_actions))

      action_mask = tf.placeholder(tf.float32, shape=(batch_size, num_actions))

      # reward from the database
      target_reward_placeholder = tf.placeholder(tf.float32, shape=(batch_size,num_actions))

      # q_action is the action picked by the network given an image
      q_action = q_network(c_frame_placeholder, action_mask, num_actions)
     
      # true_reward is the reward from sending an image and action mask through the network
      true_reward = q_network(s_t_placeholder, action_mask, num_actions, reuse=True)

      loss = tf.nn.l2_loss(target_reward_placeholder-true_reward)

      train_op = tf.train.RMSPropOptimizer(learning_rate=1e-5, momentum=0.95).minimize(loss)
      
      saver = tf.train.Saver(tf.all_variables())

      variables = tf.all_variables()
      init      = tf.initialize_all_variables()
      sess      = tf.Session()
      sess.run(init)

      epsilon = 1

      x_t = env.reset() # reset the environment
      x_t = preprocess(x_t)

      # copy it 4 times
      s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=2)
      s_t = np.expand_dims(s_t, 0)
      
      step = int(sess.run(global_step))

      ones_action = np.asarray(np.split(np.ones(num_actions*batch_size), batch_size))
      zeros_action = np.asarray(np.split(np.zeros(num_actions*batch_size), batch_size))

      while True:
         
         # total reward for the 4 actions taken
         total_reward = 0

         # render environment to screen
         env.render()

         # choose action using e-greedy: choose randomly with probability epsilon
         # step*batch_size*4 gives us how many frames we've gone over total
         epsilon = linear_annealing(step*batch_size*4, 1000000, 1, 0.1)
         r = np.random.rand()
         r = 10
         if r < epsilon:
            a_t = np.random.randint(num_actions)
         else:
            # use Q function to pick action - send in last 4 frames from history
            # send in image, get actions back, pick argmax
            # when starting the game, s_t will be the same first 4 frames
            a_t = np.argmax(sess.run([q_action], feed_dict={c_frame_placeholder:s_t, action_mask:ones_action}))
         
         # execute action (4 times!)
         for _ in range(history_length):
            env.render()
            x_t1, r_t, terminal, info = env.step(a_t) # take a step and get reward from action
            s_t1 = preprocess(x_t1) # preprocess image
            try: s_t1 = np.concatenate((s_t1, preprocess(x_t1)), axis=2) # preprocess image
            except: s_t1 = preprocess(x_t1)
            total_reward += r_t

         #s_t1 = np.expand_dims(s_t1, 0)
         # store
         e = np.asarray([s_t, a_t, total_reward, s_t1, terminal])

         # if db is over one million, replace randomly
         if len(experience_replay) > 1000000:
            idx = random.randint(0,99999)
            experience_replay[idx] = e
         else:
            experience_replay.append(e)

         # sample randomly from experience replay ---- this is what you train on!
         data = np.asarray(random.sample(experience_replay, batch_size))
         batch_s_j    = []
         batch_s_j1   = []
         batch_reward = []
         batch_action = []

         # something stupid going on her, isn't (32,84,84,4), is for some reason (32,)
         for e in data:
            batch_s_j.append(e[0])
            batch_action.append(e[1])
            batch_s_j1.append(e[3])
            batch_reward.append(e[2])

         '''
            since sending a batch through, have to go through each one after and see if
            we're using r_j as the reward or the target_reward
         '''
         # send s_j through network, use highest action and predict reward
         # minimize the reward you got with the actual reward
         # y_j is the target reward!
         # send s_j1 to the network and take the highest reward -> getting future expected reward
         target_reward = sess.run([true_reward], feed_dict={s_t_placeholder:batch_s_j1, action_placeholder:ones_action})
         print 'target_reward: ', target_reward
         exit()
         y_j = r_j + gamma*target_reward

         target_reward_vec = np.zeros(num_actions)
         target_reward_vec[a_j] = target_reward
         target_reward_vec = np.expand_dims(target_reward_vec, 0)

         action = np.zeros(num_actions)
         action[a_j] = 1
         action = np.expand_dims(action,0)

         actual_y = sess.run([actual_reward], feed_dict={s_t_placeholder:s_j, action_placeholder:action})[0][0]

         _, loss_ = sess.run([train_op, loss], feed_dict={target_reward_placeholder:target_reward_vec, action_placeholder:action, s_t_placeholder:s_j})

         if step % 10 == 0:
            print 'step:',step,'loss:',loss_

         if step % 1000 == 0:
            print 'saving model...'
            saver.save(sess, 'models/checkpoint', global_step=global_step)

         if terminal:
            x_t = env.reset() # reset the environment
            s_t = preprocess(x_t)

         step += 1


if __name__ == '__main__':

   env = gym.make('Breakout-v0')

   num_actions = int(str(env.action_space).split('(')[-1].split(')')[0])
   batch_size = 32

   experience_replay = []

   # how many frames to stack when sending to the network
   history_length = 4
      
   x_t = env.reset() # reset the environment
   x_t = preprocess(x_t)

   # copy it 4 times
   s_t = np.concatenate((x_t, x_t, x_t, x_t), axis=2)   
   
   # expand dims because convolutions expect batch_size to be first dimension
   #s_t = np.expand_dims(s_t, 0)

   # fill up experience_replay to be of size batch_size
   while len(experience_replay) < batch_size:
      total_reward = 0
      for i in range(history_length):
         a_t = np.random.randint(num_actions)
         x_t1, r_t, terminal, info = env.step(a_t) # take a step and get reward from action
         try: s_t1 = np.concatenate((s_t1, preprocess(x_t1)), axis=2) # preprocess image
         except: s_t1 = preprocess(x_t1)
         total_reward += r_t
     
      # expand dims because convolutions expect batch_size to be first dimension
      #s_t1 = np.expand_dims(s_t1, 0)

      # s_t and s_t1 are both (84,84,4)
      e = np.asarray([s_t, a_t, total_reward, s_t1, terminal])
      experience_replay.append(e)

   train(env, batch_size, num_actions, experience_replay, history_length)


