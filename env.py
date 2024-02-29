import retro
import numpy as np
import gym
import cv2

class SMBNES():
    def __init__(self, rom_name = 'SuperMarioBros-Nes'):
        
        self.frameskips = 4
        
        self.action_space = [[0,0,0,0,0,0,1,0,0],
                            [0,0,0,0,0,0,0,1,0],
                            [0,0,0,0,0,0,1,0,1],
                            [0,0,0,0,0,0,0,1,1]]
        
        self.epsilon = 0.05
        self._env = None
        self.rom_name = rom_name
        
    def run_env(self,function, *arg):
        self._env = retro.make(self.rom_name)
        try:
            return function(*arg)
        finally:
            self._env.close()
  
    def preprocess(self, action, env, frameskips):
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
        return np.dstack(state_output).astype('float16'), reward, d, info #For frame stacking
    
    def collect_episode(self, model, preprocessor):
        #Create video object. We will write the frames to this object.
        #out = cv2.VideoWriter(output_name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60/frameskips, (1800,1440))
        #You have to reset the environment to the start of the level.
        states = []
        rewards = []

        s = self._env.reset()

        #Prime the frame stack.
        s, episode_return, done, info = preprocessor(self.action_space[0], self._env, self.frameskips)
        episode_return = 0
        return_list = []
        done = False
        lives = int(info['lives'])
        
        #Enter the episode loop.
        while not(done):

            #Choose epsilon greedy action.
            if np.random.random() < self.epsilon:
                a = np.random.choice(np.arange(len(self.action_space)))
            else: 
                a_probs = model(np.expand_dims(s,0), training=False)
                a = np.argmax(a_probs[0]) 

            #Collect information on next state.
            s, reward, done, info = preprocessor(self.action_space[a], self._env, self.frameskips)
            raw_obs = self._env.get_screen()[:, :, [0, 1, 2]]
            states.append([s,raw_obs])
            rewards.append(reward)
            return_list.append(episode_return)

        #Release the video object. 
        #out.release()
        #clear_output()
        print ('Evaluation complete.')
        return states, rewards