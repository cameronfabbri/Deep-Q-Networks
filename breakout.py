import gym
#import tensorflow as tf
from collections import deque
import cv2
from tqdm import tqdm

# want it to keep playing after the game finished i.e start a new one

def train():

   while True:
      env.render()
      action = env.action_space.sample()
      print action
      state, reward, done, info, = env.step(action)

      if done:
         game_num += 1
         print 'Game number: ', game_num
         observation = env.reset()

if __name__ == '__main__':
   game        = 'Breakout-v0'  # name of the game
   game_num    = 1              # always start at first game
   num_actions = 6              # number of actions for the game
   D_size      = 100            # size of the database (number of steps)
   SIZE        = (210, 160)     # size of the screen input
   env         = gym.make(game) # create the game
   play_random = 200            # number of steps to play randomly
   seq_length  = 4              # length of sequence for input

   experience_list = []

   print 'Game: ', game
   print 'Game number: ', game_num
   print

   print 'Randomly playing to fill database'
   print

   # first play randomly to initialize database to sample from
   while True:
      initial_observation = env.reset() # first image/state
      for step in tqdm(range(play_random)):
         env.render() # render environment
         action = env.action_space.sample() # get random action

         # execute the random action in the environment, get results
         # current state is the state AFTER taking an action (s_t+1)
         current_state, reward, done, info = env.step(action)
     
         # if this is its first step then the init observation is the previous state
         if step == 0:
            experience = [initial_observation, action, reward, current_state]

         # if it's not the first step then the PREVIOUS state is the previous state (duh)
         else:
            # this is correct because it grabs the [-1] from the experience list, aka the
            # 'current_state' after completing the last action
            previous_state = experience_list[step-1][-1]
            experience = [previous_state, action, reward, current_state]

         # in case it ends before we are done filling our queue
         if done:
            env.reset()

         experience_list.append(experience)

      break

   # faster than python list class
   replay_database = deque(experience_list)

   print 'Done'

   exit()
