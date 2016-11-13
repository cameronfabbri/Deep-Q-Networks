import gym
from collections import deque
import cv2
from tqdm import tqdm
from random import randint
import tensorflow as tf
import architecture as arch

def get_feed_dict(batch_size, images_placeholder, seq_length, replay_database):

   # get the size of the database
   replay_size = len(replay_database)

   start = randint(0, replay_size-seq_length)

   feed_dict = {
      images_placeholder: original_images,
   }

   return feed_dict


'''
   Training will sample randomly
   Future updates: take in boolean gray as an input so the network
   is modular, aka can train on color or gray

   starting with batch size 1 just so I can understand all the parts

   After it'll be sending in batch_size*seq_length into the network

'''
def train(replay_database, seq_length, SHAPE):


   # set up computational graph
   with tf.Graph().as_default():
      global_step = tf.Variable(0, name='global_step', trainable=False)
      
      # tensor of size batch, width, height, channels*seq_length
      images_placeholder = tf.placeholder(tf.float32, shape=(batch_size, SHAPE[1], SHAPE[0], 3*seq_length)) 
      next_step_image    = tf.placeholder(tf.float32, shape=(1, SHAPE[1], SHAPE[0], 3))

      # Q values for each action
      target_value = architecture.inference(images_placeholder, 'train')
      actual_value = architecture.inference(next_step_image, 'train')

      loss = architecture.loss(target_value, actual_value)
      
      '''
         grab random batch -> replay database must be of size larger than batch_size*seq_length
         pick random start spot. Ensures it does not extend the db and also only picks things
         once per iteration
      '''

      while True:
         # start a new game
         initial_observation = env.reset()

         '''
           The input is the current state image
           The output is the predicted Q value for each action
           Loss: Net(s_i)(a_i) - (r_i+1 max(Net(s_i+1)))
         '''

         # get sequence and run through network to get max action
         feed_dict = get_feed_dict(batch_size, images_placeholder, seq_length, replay_database)

         

         action = env.action_space.sample()
         state, reward, done, info, = env.step(action)

         #if done:
         #   game_num += 1
         #   print 'Game number: ', game_num
         #   observation = env.reset()

if __name__ == '__main__':

   '''
      General options
      Maybe these should be parameters or something in the future...fine for now
   '''
   game        = 'Breakout-v0'  # name of the game ... heh
   game_num    = 1              # always start at first game
   num_actions = 6              # number of actions for the game
   SIZE        = (84, 84)     # NEW size of the screen input (if RGB don't need 3rd dimension)
   env         = gym.make(game) # create the game
   play_random = 20            # number of steps to play randomly
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

      for step in tqdm(range(play_random)):
      #for step in range(play_random):
         env.render() # render environment
         
         #action = env.action_space.sample(seed=ranint(0,100)) # get random action
         action = randint(0, num_actions-1)

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

   '''
   just testing the database

   i = 0
   for experience in replay_database:
      print 'previous state: ', experience[0].shape
      print 'action: ', experience[1]
      print 'reward: ', experience[2]
      print 'current_state: ', experience[3].shape
      exit()
   '''

   print 'Done with filling database'
   train(replay_database, seq_length, SHAPE, batch_size) 
