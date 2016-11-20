import gym
from collections import deque
import cv2
from tqdm import tqdm
from random import randint
import tensorflow as tf
import architecture
import numpy as np
import random

def getFeedDict(batch_size, state_t_p, action_p, seq_length, replay_database, predict, SHAPE):

   # predict determines if the feed dictionary is being used for prediction or not.
   # if it is, then don't put the action in the feed_dict

   # get the size of the database
   replay_size = len(replay_database)

   start = randint(0, replay_size-(seq_length*batch_size))
   end = (start + (seq_length*batch_size))

   # list containing the stacked sequences. (batch_size, SIZE[0], SIZE[1], seq_length)
   states = []

   all_states = np.asarray([replay_database[_][0] for _ in range(start, end)])
   
   split_states = np.asarray(np.split(all_states, seq_length))
   split_states = np.reshape(split_states, (batch_size, SHAPE[0], SHAPE[1], seq_length))

   for s in split_states:
      states.append(s)
   states = np.asarray(states)

   if predict:
      actions   = np.asarray(replay_database[end-1][1])
      feed_dict = {
         state_t_p: states,
         action_p: actions
      }
   else:
      feed_dict = {
         state_t_p: states
      }

   return feed_dict

'''
   Takes in the two choices and a probability for each choice

   choices: choose a random action, or select max(action) from running the network
            [0,1]: 0 for random aciton, 1 for running the network

   probabilities: initially start at 100% probability to choose random, then decay to 0.1
            [0.0, 1.0]
'''
def randomChoice(choices, probabilities):
   x = random.uniform(0,1)
   cum_prob = 0.0
   
   for choice, prob in zip(choices, probabilities):
      cum_prob += prob
      if x < cum_prob:
         break
   return choice


'''
   Perform inference on what's currently going on in the emulator and put that
   in the database.

   Grab a random 4 sequence from the database. Get the action from the last state
   in that sequence.

   Send the sequence through the network, and get the value that the last action
   outputs.

   Get the maximum value from sending the next state through the network.

   loss = Net(S_i)[a_i] - (gamma + max(Net(s_i+1)))

   Copy the weights of the network to the one running inference every X steps
'''
def train(checkpoint_dir, replay_database, seq_length, SHAPE, batch_size, gamma, rand, num_actions):

   game_num = 1

   initial_rand = rand[0]
   final_rand   = rand[1]

   # start rand at the init, then decay it to final rand
   rand = initial_rand

   # set up computational graph
   with tf.Graph().as_default():
      global_step = tf.Variable(0, name='global_step', trainable=False)
      
      state_t_p    = tf.placeholder(tf.float32, shape=(batch_size, SHAPE[1], SHAPE[0], seq_length)) 
      next_state_p = tf.placeholder(tf.float32, shape=(batch_size, SHAPE[1], SHAPE[0], seq_length)) 
      action_p     = tf.placeholder(tf.float32, shape=(batch_size, 1))

      # first choose an action to take - observe reward and image - update weights every X steps
      action_choice = architecture.inference(state_t_p)

      # predict what the action taken will give you by returning its output
      predicted_value = architecture.predict(state_t_p, action_p)
      
      # get the actual value the action you took was by taking the max of all values returned
      #actual_value = architecture.train(next_state_p)
      actual_value = architecture.train(state_t_p)

      # get the loss between the predicted value and the actual value from the action
      loss = architecture.loss(predicted_value, actual_value, gamma)

      train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss, global_step=global_step)

      variables = tf.all_variables()
      init      = tf.initialize_all_variables()
      sess      = tf.Session()

      try:
         os.mkdir(checkpoint_dir)
      except:
         pass

      sess.run(init)
      print '\nRunning session...\n'

      saver = tf.train.Saver(variables)

      tf.train.start_queue_runners(sess=sess)

      graph_def = sess.graph.as_graph_def(add_shapes=True)
      summary_writer = tf.train.SummaryWriter(checkpoint_dir+"training", graph_def=graph_def)

      step = int(sess.run(global_step))

      # this just grabs the first 4 frames of the game and puts them in s_t for use on the first step
      i = 0
      s_t = []
      while i <= 4:
         # just starting a game
         if i == 0:
            state = env.reset()
            env.render()
            i += 1
         else: # adding 3 more to the state
            action = randint(0, num_actions-1)
            state = env.step(action)[0]
            state = cv2.resize(state, SHAPE, interpolation=cv2.INTER_CUBIC)
            state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
            s_t.append(state)
            i += 1

      s_t = np.asarray(s_t)
      s_t = np.swapaxes(s_t, 0, 2)
      s_t = np.expand_dims(s_t, 0)
      # now s_t contains a sequence of 4 states
      
      while True:
         env.render()
         step += 1
         print 'Step:', step

         # IMPORTANT - THIS SHOULD DECAY - SETTING TO 75% RAND FOR NOW
         # determine whether or not to choose a random action or use the network
         # choice = randomChoice([0,1], [initial_rand, 1-initial_rand])
         # choice = randomChoice([0,1], [0.75, 0.25]) 75% rand, 25% inference
         choice = randomChoice([0,1], [0, 1]) # 100% inference for testing

         # choose random action
         if choice == 0:
            action = randint(0, num_actions-1)
         else: # do NOT choose a random action, run inference on the current sequence. THIS ISN'T TRAIN!!
            action_values = np.asarray(sess.run([action_choice], feed_dict={state_t_p:s_t})[0][0])
            action = np.argmax(action_values) # get argmax action from values


         # execute that action FOUR TIMES (or seq_length times)
         s_t1   = []
         reward = 0
         for i in range(seq_length):
            s_t1_, r, terminal, info = env.step(action) # take a step in the environment
            s_t1_ = cv2.resize(s_t1_, SHAPE, interpolation=cv2.INTER_CUBIC)
            s_t1_ = cv2.cvtColor(s_t1_, cv2.COLOR_RGB2GRAY)

            s_t1.append(s_t1_)
            reward += r

         s_t1 = np.asarray(s_t1)
         experience = [s_t, action, reward, s_t1]
         replay_database.append(experience) # add to experince db

         # set s_t to the current timestamp because the next time in the loop it will be the prev
         s_t = s_t1

         # NOW TRAIN
         # sample minibatch from database - this has NOTHING to do with what's on screen
         feed_dict = getFeedDict(batch_size, state_t_p, action_p, seq_length, replay_database, False, SHAPE)
         
         # send it a batch of states to the network
         pred_val = sess.run([predicted_value], feed_dict=feed_dict)
         act_val  = sess.run([actual_value], feed_dict=feed_dict)
         print pred_val
         print act_val
         exit()

         '''
            for tomorrow:
               - get the decaying random working
               - follow rest of algorithm
               - remember to do same action for seq_length


         '''

         #state, reward, terminal, info = env.step(action)

         if terminal:
            game_num += 1
            print 'Game number: ', game_num
            initial_observation = env.reset()

if __name__ == '__main__':

   '''
      General options
      Maybe these should be parameters or something in the future...fine for now
   '''
   game         = 'Breakout-v0'  # name of the game ... heh
   game_num     = 1              # always start at first game
   num_actions  = 6              # number of actions for the game
   SHAPE        = (84, 84)       # NEW size of the screen input (if RGB don't need 3rd dimension)
   env          = gym.make(game) # create the game
   play_random  = 10            # number of steps to play randomly
   seq_length   = 4              # length of sequence for input
   grayscale    = True           # whether or not to use grayscale images instead of rgb
   batch_size   = 1              # size of the batch. Network will receive size batch_size*seq_length*dims
   gamma        = 0.99           # Decay rate of future rewards
   initial_rand = 1.0            # starting probability that you will pick a random action
   final_rand   = 0.1            # final probability that you will pick a random action

   checkpoint_dir = 'models/'

   rand = [initial_rand, final_rand]

   # the number of viable actions to take given the loaded game
   num_actions = int(str(env.action_space).split('(')[-1].split(')')[0])
   
   # initialize a list for storing experience
   experience_list = []

   print 'Game: ', game
   print 'Game number: ', game_num
   print 'Input size: ', SHAPE
   print 
   print 'Randomly playing to fill database'
   print

   # first play randomly to initialize database to sample from
   while True:
      initial_observation = env.reset() # first image/state

      # resize the image
      initial_observation = cv2.resize(initial_observation, SHAPE, interpolation=cv2.INTER_CUBIC)

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
         current_state, reward, terminal, info = env.step(action) # take a step in the environment

         # resize image
         current_state = cv2.resize(current_state, SHAPE, interpolation=cv2.INTER_CUBIC)

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
            previous_state = cv2.resize(experience_list[step-1][-1], SHAPE, interpolation=cv2.INTER_CUBIC)

            # previous state should already be gray since it's coming from the last step

            experience = [previous_state, action, reward, current_state]

         # in case it ends before we are terminal filling our queue
         if terminal:
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
   
   print
   print 'Done with filling database'
   print
   print 'Replay database size: ', len(replay_database)
   print

   train(checkpoint_dir, replay_database, seq_length, SHAPE, batch_size, gamma, rand, num_actions)
