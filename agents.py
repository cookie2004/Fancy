import models
import env

import time
import threading
import logging
import retro
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
import numpy as np
import cv2
import gym
import matplotlib.pylab as plt
from IPython.display import clear_output

import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

import multiprocessing
from multiprocessing import Process
 
                 
class Agent():
    def __init__(self,runname, rom_name = 'SuperMarioBros-Nes'):

        self.rom_name = rom_name

        self.env = retro.make(rom_name)
        
        self.action_space = [[0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,1,0,1],
                            [0,0,0,0,0,0,0,1,1]]
     
        # Every number of steps equal to the epoch length, evalulation a greedy run of the Q function.
        self.eval_epsilon = 0.05     #Epsilon value that the NN will be evaluated at in the evaluate_NN function.
        self.eval_reps = 1           #Number of evaluation replicates. For SMB, set this to one or training will take a long time.
        self.save_freq = 5000        #Epoch length is the number of steps before history data is saved.
        self.lives = 2               #For SMB, this number should be 2. 
        self.len_of_episode = 10000  #Max number of steps per episode. 
        self.steps_taken = 0         #This will track the total number of steps so the appropriate epsilon value can be calculated. 
        self.num_episodes = 0
        self.runname = runname
        self.steps_per_report = 1000
        self.epsilon = 1.0           #Class storage for epsilon.
        self.epsilon_max = 0.95       #Max epsilon in traning. 
        self.epsilon_min = 0.1       #Minimum epsilon in training.
        self.epsilon_lag = 10000     #Number of steps epsilon will stay at the max at the start of training.
        self.annealing_time = 1000000#Number of steps it will take to go from epsilon_max to epsilon_min.
        self.gamma = 0.99            #Discount rate
        self.max_memory_len = 30000  #Max number of frame data to keep in storage.
        self.batch_size = 32         #Number of frames to pull from memory and perform gradient descent. 
        self.steps_per_update = 4    #Number of steps between gradient descent
        self.reward_scaler = 1       #Scales the reward recieved by the environment.
        self.target_update = 10000   #Number of steps before behavior model weights are copied to target model. 
        self.window = 25             #Number of frames averaged in the sliding window function.
        self.frameskip = 12          #Total stepped frames. Frame stack is constructed from linearly spaced frames. 
        self.framestack = 4
        self.termination_penalty = 10#Absolute value of the negative reward value for loosing a life or dying. 
        self.clipping = False        #Enable reward clipping?
        self.episodes_per_eval = 10  #Number of steps before evaluate_NN
        
        #Initialize containers which will be prepared in thread_prep()
        self.loss_history = []
        self.action_history = []
        self.state_history= []
        self.next_state_history = []
        self.reward_history = []
        self.done_history = []
        self.episodic_return = []
        self.return_history = [] 
        self.epsilon_schedule = []
        self.grads = []
        self.evaluations = []

        #Initialize target and behavior network.
        self.seed = 42
        self.behavior = models.CONV_ANN(self.action_space, self.framestack, 84) #RAM_ANN(self.action_space, 128, self.frameskip, self.seed)
        self.target = models.CONV_ANN(self.action_space, self.framestack, 84) #RAM_ANN(self.action_space, 128, self.frameskip,self.seed)
    
    #Rewards are either -1, 0, or 1 at each timestep. If clipping is disabled, the full reward from the environment is given.
    def clip_reward(self, reward):
        """Clips the received reward to a specified range.

        This function limits the reward value to a maximum of 1, a minimum of -1,
        or leaves it unchanged depending on the `clipping` attribute. This can
        be useful for stabilizing and normalizing rewards in reinforcement learning.

        Args:
            reward (float): The original reward value to be clipped.

        Returns:
            float: The clipped reward value, either -1, 0, or 1.
        """
        if not self.clipping:
            return reward
        else:
            if reward > 0:
                return 1
            elif reward == 0:
                return 0
            else:
                return -1
    
    #The Conv NN analyzes 4 stacked frames at a time. Thus, you need to create a sliding frame. As a new frame comes in, the oldest frame is removed.
    def popback(self, state_block, incoming_state):
        state_block.pop(0)
        state_block.append(incoming_state)
        return state_block

    #Gradient descent applied to experiences stored in memory.
    def gradient_update(self, 
                        runname,
                        state_history, 
                        next_state_history,
                        rewards_history,
                        action_history,
                        loss_history,
                        behavior_model,
                        target_model,
                        gamma,
                        batch_size,
                        done_history,
                        action_space):
        """Performs a gradient update step for Q-learning in a reinforcement learning agent.

        This function samples experiences from a replay buffer, calculates Q-values using
        both the behavior and target models, computes the loss based on Q-learning updates,
        and applies gradients to the behavior model's network to improve its action-value
        predictions.

        Args:
            runname (str): Name for identifying data saved during training.
            state_history (list): List of previous states experienced by the agent.
            next_state_history (list): List of states that followed those in state_history.
            rewards_history (list): List of rewards received after each state in state_history.
            action_history (list): List of actions taken in each state in state_history.
            loss_history (list): List of loss values from previous gradient updates.
            behavior_model (object): The model used for action selection.
            target_model (object): The model used for stabilizing training.
            gamma (float): Discount factor for future rewards.
            batch_size (int): Number of experiences to sample for each update.
            done_history (list): List of booleans indicating episode terminations.
            action_space (int): Number of possible actions the agent can take.

        Returns:
            list: The updated loss_history with the latest loss value appended.
        """
        # Get indices of samples for replay buffers
        indices = np.random.choice(range(len(done_history)), size=batch_size)
        # Using list comprehension to sample from replay buffer
        state_sample = np.array([state_history[i] for i in indices])
        next_state_sample = np.array([next_state_history[i] for i in indices])
        rewards_sample = [rewards_history[i] for i in indices]
        action_sample = [action_history[i] for i in indices]
        done_sample = tf.convert_to_tensor([float(done_history[i]) for i in indices])
        future_rewards = target_model.model.predict(next_state_sample, verbose=0)
        updated_q_values = rewards_sample + gamma * tf.reduce_max(future_rewards, axis=1)
        updated_q_values = updated_q_values *(1-done_sample) - done_sample*self.termination_penalty

        masks = tf.one_hot(action_sample, len(action_space))
        with tf.GradientTape() as tape:  
            q_values = behavior_model.model(state_sample)
            q_actions = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = behavior_model.loss_fn(updated_q_values, q_actions)
        loss_history = loss_history.append(loss)
        grads = tape.gradient(loss, behavior_model.model.trainable_variables)
        behavior_model.model.optimizer.apply_gradients(zip(grads, behavior_model.model.trainable_variables))
        

    #Save all histories. Memory storage is inefficent at the moment because many of the frames in the 4-stacked frames for the NN input overlap. 
    #Data arragement needs to be changed so only unique frames are saved. This would reduce the memory consumption by 75%. 
    def save_history(self,):   
        runname = self.runname
        np.save(runname + 'action_history',self.action_history)
        np.save(runname + 'state_history', self.state_history)
        np.save(runname + 'next_state_history', self.next_state_history)
        np.save(runname + 'reward_history', self.reward_history)
        np.save(runname + 'done_history', self.done_history)
        np.save(runname + 'return_history', self.episodic_return)
        np.save(runname + 'evaluations', self.evaluations)
        np.save(runname + 'loss_history', self.loss_history)
        self.behavior.model.save(runname+'_behavior')
        self.target.model.save(runname+'_target')

    #Function incomplete.   
    def load_history(self,runname):   
        self.action_history = [i for i in np.load(runname + 'action_history.npy', allow_pickle=True)]
        self.state_history = [list(i) for i in np.load(runname + 'state_history.npy', allow_pickle=True)]
        self.next_state_history = [list(i) for i in np.load(runname + 'next_state_history.npy', allow_pickle=True)]
        self.reward_history = [i for i in np.load(runname + 'reward_history.npy', allow_pickle=True)]
        self.done_history = [i for i in np.load(runname + 'done_history.npy', allow_pickle=True)]
        self.return_history = [i for i in np.load(runname + 'return_history.npy', allow_pickle=True)]
        
        #You have to convert the numpy arrays to lists.
        self.loss_history = list(np.load(runname + 'loss_history.npy', allow_pickle=True))
        self.evaluations = list(np.load(runname + 'evaluations.npy', allow_pickle=True))
        
        
        self.behavior.toymodel = keras.models.load_model(runname+'_behavior')
        self.target.toymodel = keras.models.load_model(runname+'_target')
    
    #The raw observation from the environment is preprocessed by grayscaling and shrinking the image to 84x84 pixel. 
    def preprocess(self, action, env, framestacks, frameskips):
        """Preprocesses multiple game frames for use in a reinforcement learning agent.

        This function takes an action, executes it in the environment for a specified
        number of frames, concatenates the resulting frames, and returns the preprocessed
        state, accumulated reward, done flag, and environment info.

        Args:
            action (int): The action to be executed in the environment.
            env (object): The environment object to interact with.
            frameskips (int): The number of frames to skip and concatenate.

        Returns:
            tuple: (state, reward, done, info)
                - state (numpy.ndarray): The preprocessed state, stacked along the depth
                  axis and normalized to the range [0, 1].
                - reward (float): The accumulated reward over the skipped frames.
                - done (bool): A flag indicating whether the episode has terminated.
                - info (dict): Additional information from the environment.
        """
        state_output = []
        reward = 0
        for i in range(frameskips):
            s, r, d, info = env.step(action)
            reward += r
            s = cv2.cvtColor(s, cv2.COLOR_RGB2GRAY)
            s = cv2.resize(s, (84, 84), interpolation=cv2.INTER_AREA)
            state_output.append(s/255.0)
        stack = np.linspace(0,len(state_output), framestacks, dtype=int, endpoint=False)

        state_output = [state_output[i] for i in stack]
        return np.dstack(state_output).astype('float16'), reward, d, info #For frame stacking
    
    #Ensures memory doesn't get too large. 
    def memory_manager(self,array, mem_size):  
        """Manages memory usage by removing elements from an array when it exceeds a specified size.

        This function ensures that the array's length does not grow beyond the specified memory
        limit. It accomplishes this by deleting older elements from the beginning of the array
        when necessary.

        Args:
            array (list): The array to be managed.
            mem_size (int): The maximum allowed length of the array.

        Returns:
            None
        """
        num_delete = len(array) - mem_size
        if num_delete < 0:
            None
        else:
            del array[:num_delete]
    
    #Calculates the epsilon based on input parameters. 
    def piecewise_epsilon(self, steps_taken, lag, annealingtime, ep_min, ep_max): 
        """Returns an epsilon value that decreases over time using a piecewise linear schedule.

        This function implements a piecewise linear schedule for epsilon decay in reinforcement
        learning. It maintains a high epsilon value for an initial lag period, then gradually
        decreases it over an annealing time, and finally holds it at a minimum value.

        Args:
            steps_taken (int): The number of steps taken in the environment so far.
            lag (int): The initial period during which epsilon remains at its maximum value.
            annealingtime (int): The duration of the linear decay phase.
            ep_min (float): The minimum epsilon value to reach after annealing.
            ep_max (float): The initial maximum epsilon value.

        Returns:
            float: The epsilon value to use for the current step, calculated based on the
                   piecewise linear schedule.
        """
        anneal_slope= (ep_min-ep_max)/(lag+annealingtime-lag)
        if steps_taken < lag: return ep_max
        if (steps_taken >= lag) and (steps_taken < (lag+annealingtime)): return anneal_slope*steps_taken+(ep_max-anneal_slope*lag)
        else: return ep_min

    #Episodic return is noisy. It is easier to visualize learning progress by calculating the average return over several episodes. 
    def sliding_average(self, array, n):
        """Calculates the sliding average of an array over a window of size n.

        This function smooths out noisy data by computing the average of n consecutive
        elements in the array, sliding the window along the array's length. It's useful
        for visualizing trends in episodic returns in reinforcement learning.

        Args:
            array (list): The array of values to smooth.
            n (int): The size of the sliding window.

        Returns:
            list: The smoothed array, containing the sliding averages for each valid window.
        """
        output = []
        for i in range(len(array)):
            try:
                output.append(np.average(array[i:i+n]))
            except IndexError:
                break
        return output
    
   
    #Visualize progress by analyzing episodic return, sliding window, evaluation returns, loss, and epsilon. 
    def plot_data(self,):
        clear_output()
        fontsize=7.5
        linewidth = 1
        fig = plt.figure(figsize=(7.5, 6))
        gs = GridSpec(nrows=5, ncols=2)
        #fig, ax = plt.subplots(5,2,figsize=(5,7))

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(self.episodic_return, linewidth=linewidth)
        ax0.set_xlabel('Episode',fontsize=fontsize)
        ax0.set_ylabel('Return',fontsize=fontsize)

        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(self.sliding_average(self.episodic_return, self.window), linewidth=linewidth)
        ax1.set_xlabel('Episode',fontsize=fontsize)
        ax1.set_ylabel('Sliding Return',fontsize=fontsize)

        ax2 = fig.add_subplot(gs[2, 0])
        ax2.plot(self.evaluations, linewidth=linewidth)
        ax2.set_xlabel('Episode',fontsize=fontsize)
        ax2.set_ylabel('Eval Return',fontsize=fontsize)

        ax3 = fig.add_subplot(gs[3, 0])
        ax3.plot(self.loss_history, linewidth=linewidth)
        ax3.set_xlabel('Training Step',fontsize=fontsize)
        ax3.set_ylabel('Loss',fontsize=fontsize)
        ax3.set_yscale('log')

        ax4 = fig.add_subplot(gs[4, 0])
        Y = [self.piecewise_epsilon(j, self.epsilon_lag, self.annealing_time, self.epsilon_min, self.epsilon_max) for j in np.arange(self.steps_taken)]
        ax4.plot(Y, linewidth=linewidth)
        ax4.set_xlabel('Steps taken',fontsize=fontsize)
        ax4.set_ylabel('Epsilon',fontsize=fontsize)

        ax0.tick_params(axis='both', which='major', labelsize=7.5)
        ax1.tick_params(axis='both', which='major', labelsize=7.5)
        ax2.tick_params(axis='both', which='major', labelsize=7.5)
        ax3.tick_params(axis='both', which='major', labelsize=7.5)
        ax4.tick_params(axis='both', which='major', labelsize=7.5)

        plt.tight_layout()
        plt.show()


    def evaluate_NN(self, env, epsilon, replicates=1, run_fancy=False):
        """Evaluates the performance of a neural network (NN) agent in a given environment.

        This function runs multiple evaluation episodes using epsilon-greedy action selection,
        calculates the average return across those episodes, and appends it to a list of
        evaluations for tracking progress. It handles multiple lives within an episode
        by continuing using the fire button when a life is lost.

        Args:
            env (object): The environment to evaluate the agent in.
            epsilon (float): The probability of taking a random action during evaluation.
            replicates (int): The number of evaluation episodes to run.

        Returns:
            None - Updates evaluation container.
        """
        av_returns = []
        for i in range(replicates):
            s = env.reset()
            s, episode_return, done, info = self.preprocess(self.action_space[1], env, self.framestack, self.frameskip)
            episode_return = 0
            done = False
            lives = int(info['lives'])
            while True:
                
                #Choose epsilon greedy action.
                if np.random.random() < epsilon:
                    a = np.random.choice(np.arange(len(self.action_space)))
                else: 
                    a_probs = self.target.model(np.expand_dims(s,0), training=False)
                    a = tf.argmax(a_probs[0]).numpy()
                
                #Collect information on next state.
                s, reward, done, info = self.preprocess(self.action_space[a], env, self.framestack, self.frameskip)
                episode_return += reward

                if not (int(info['lives']) == lives):                                         
                    lives = int(info['lives'])
                    s, reward, done, info = self.preprocess(self.action_space[0], env, self.framestack, self.frameskip)
                    av_returns.append(episode_return)  
                    episode_return = 0 
                    break
                if done:
                    av_returns.append(episode_return)   
                    break   
        self.evaluations.append(np.average(av_returns))


    def save_video(self, env, filename='output.avi'):

        result = cv2.VideoWriter(filename,  
                                 cv2.VideoWriter_fourcc(*'MJPG'), 
                                 60/self.frameskip, (240, 224)) 

        s = env.reset()
        s, episode_return, done, info = self.preprocess(self.action_space[1], env, self.framestack, self.frameskip)
        episode_return = 0
        done = False
        lives = int(info['lives'])
        while True:
            result.write(env.get_screen())
            #Choose epsilon greedy action.
            if np.random.random() < self.epsilon:
                a = np.random.choice(np.arange(len(self.action_space)))
            else: 
                a_probs = self.target.model(np.expand_dims(s,0), training=False)
                a = tf.argmax(a_probs[0]).numpy()
            #Collect information on next state.
            s, reward, done, info = self.preprocess(self.action_space[a], env, self.framestack, self.frameskip)
            episode_return += reward

            if not (int(info['lives']) == lives):                                         
                episode_return = 0 
                break
            if done:

                break   


        result.release()

    #Collect a set number of training steps. 
    def episode(self, num_training_steps, epsilon):    
        
        epsilon = self.piecewise_epsilon(self.steps_taken, 
                                         self.epsilon_lag, 
                                         self.annealing_time, 
                                         self.epsilon_min, 
                                         self.epsilon_max)
        steps = 0
        while steps < num_training_steps:
            
            if self.num_episodes%self.episodes_per_eval == 0:
                self.evaluate_NN(self.env,self.eval_epsilon, self.eval_reps) 
     
            episode_step = 0
            
            epi_return = 0 
            self.env.reset()
            
            s, reward, done, info = self.preprocess(self.action_space[1], self.env, self.framestack, self.frameskip)
            lives = int(info['lives'])
            #Enter the loop.
            while steps < num_training_steps and self.len_of_episode > episode_step:
                if self.steps_taken%self.steps_per_report==0: 
                    self.plot_data()
                                 
                if self.steps_taken%self.save_freq==0 and self.steps_taken>1: 
                    self.save_history()
                    
                #Choose an action from according to epsilson-greedy policy.  
                if np.random.random() < epsilon:
                    a = np.random.choice(np.arange(len(self.action_space)))
                else: 
                    a_probs = self.behavior.model(np.expand_dims(s,0), training=False)
                    a = tf.argmax(a_probs[0]).numpy()
                    
                s_prime, reward, done, info = self.preprocess(self.action_space[a], self.env, self.framestack, self.frameskip)
                epi_return += reward
              
                #Save to history
                self.reward_history.append(self.clip_reward(reward)*self.reward_scaler)
                self.state_history.append(s)
                self.action_history.append(a)
                self.next_state_history.append(s_prime)
                self.done_history.append(done)
                  
                if not (int(info['lives']) == lives):                                         
                    self.done_history[-1] = True 
                    self.episodic_return.append(epi_return)
                    epi_return = 0
                    s, reward, done, info = self.preprocess(self.action_space[0], self.env, self.framestack, self.frameskip)
                    break
                    
                if self.steps_taken>self.batch_size and self.steps_taken%self.steps_per_update==0:
  
                    self.gradient_update(self.runname,
                                    self.state_history, 
                                    self.next_state_history,
                                    self.reward_history,
                                    self.action_history,
                                    self.loss_history,
                                    self.behavior,
                                    self.target,
                                    self.gamma,
                                    self.batch_size,
                                    self.done_history,
                                    self.action_space)


                if self.steps_taken%self.target_update==0:
                    self.target.model.set_weights(self.behavior.model.get_weights()) 
                    print ('Target model updated!')
                s = s_prime

                steps += 1
                self.steps_taken += 1
                episode_step += 1
                
                self.memory_manager(self.action_history, self.max_memory_len)
                self.memory_manager(self.state_history, self.max_memory_len)
                self.memory_manager(self.next_state_history, self.max_memory_len)
                self.memory_manager(self.reward_history, self.max_memory_len)
                self.memory_manager(self.done_history, self.max_memory_len)
            self.num_episodes += 1 

 