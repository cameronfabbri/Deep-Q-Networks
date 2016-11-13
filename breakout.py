import gym
from collections import deque
import cv2
from tqdm import tqdm
from random import randint

'''
   Training will sample randomly
   Future updates: take in boolean gray as an input so the network
   is modular, aka can train on color or gray

   starting with batch size 1 just so I can understand all the parts

   After it'll be sending in batch_size*seq_length into the network

'''
def train(replay_database, seq_length, SHAPE):

   # Don't need to import until we're actually using them
   import tensorflow as tf
   import architecture as arch

   # set up computational graph
   with tf.Graph().as_default():
      global_step = tf.Variable(0, name='global_step', trainable=False)
      
      # tensor of size batch, width, height, channels*seq_length
      images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, SHAPE[1], SHAPE[0], 3*seq_length)) 

      # grab random batch




      while True:
         # start a new game
         initial_observation = env.reset()

         action = env.action_space.sample()
         state, reward, done, info, = env.step(action)

         if done:
            game_num += 1
            print 'Game number: ', game_num
            observation = env.reset()

if __name__ == '__main__':

   '''
      General options
      Maybe these should be parameters or something in the future...fine for now
   '''
   game        = 'Breakout-v0'  # name of the game ... heh
   game_num    = 1              # always start at first game
   num_actions = 6              # number of actions for the game
   SIZE        = (100, 100)     # NEW size of the screen input (if RGB don't need 3rd dimension)
   env         = gym.make(game) # create the game
   play_random = 200            # number of steps to play randomly
   seq_length  = 4              # length of sequence for input
   grayscale   = True           # whether or not to use grayscale images instead of rgb
   batch_size  = 1              # size of the batch. Network will receive size batch_size*seq_length*dims

   # the number of viable actions to take given the loaded game
   num_actions = int(str(env.action_space).split('(')[-1].split(')')[0])
   
   # initialize a list for storing experience
   experience_list = []

   print 'Game: ', game
   print 'Game number: ', game_num
   print 'Input size: ', SIZE
   print 
   print 'Randomly playing to fill database'
   print

   # first play randomly to initialize database to sample from
   while True:
      initial_observation = env.reset() # first image/state

      # resize the image
      initial_observation = cv2.resize(initial_observation, SIZE, interpolation=cv2.INTER_CUBIC)

      # possibly convert to gray
      if grayscale:
         initial_observation = cv2.cvtColor(initial_observation, cv2.COLOR_RGB2GRAY)

      #for step in tqdm(range(play_random)):
      for step in range(play_random):
         env.render() # render environment
         
         action = env.action_space.sample(seed=ranint(0,100)) # get random action
         #action = randint(0, num_actions)
         print action

         # execute the random action in the environment, get results
         # current state is the state AFTER taking an action (s_t+1)
         current_state, reward, done, info = env.step(action) # take a step in the environment

         # resize image
         current_state = cv2.resize(current_state, SIZE, interpolation=cv2.INTER_CUBIC)

         # possibly convert to gray
         if grayscale:
            current_state = cv2.cvtColor(current_state, cv2.COLOR_RGB2GRAY)

         # if this is its first step then the init observation is the previous state
         if step == 0:
            experience = [initial_observation, action, reward, current_state]

         # if it's not the first step then the PREVIOUS state is the previous state (duh)
         else:
            # this is correct because it grabs the [-1] from the experience list, aka the
            # 'current_state' after completing the last action
            previous_state = cv2.resize(experience_list[step-1][-1], SIZE, interpolation=cv2.INTER_CUBIC)

            # previous state should already be gray since it's coming from the last step

            experience = [previous_state, action, reward, current_state]

         # in case it ends before we are done filling our queue
         if done:
            env.reset()
            game_num += 1
            print 'Game number: ', game_num

         experience_list.append(experience)

      break

   # faster than python list class
   replay_database = deque(experience_list)

   i = 0
   for experience in replay_database:
      if i < 15:
         i += 1
         continue
      print 'previous state: ', experience[0].shape
      print 'action: ', experience[1]
      print 'reward: ', experience[2]
      print 'current_state: ', experience[3].shape

      exit()

   print 'Done with filling database'
   train(replay_database, seq_length, SHAPE, batch_size) 
