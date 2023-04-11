'''
Created on 8 Mar 2023

@author: steam
'''

import random
import numpy as np

from .td_controller import TDController

# Simplified version of the predictor from S+B

class QLearner(TDController):
    '''
    classdocs
    '''

    def __init__(self, environment):
        TDController.__init__(self, environment)

    def initialize(self):
               
        # Set up experience replay buffer
        TDController.initialize(self)
        
        # Change names to change titles on drawn windows
        self._v.set_name("Q-Learning Expected Value Function")
        self._pi.set_name("Q-Learning Greedy Policy")
            
    def _update_action_and_value_functions_from_episode(self, episode):
        
        steps = episode.number_of_steps()
        for step in range(steps-1):
            # extract information from step
            state = episode.state(step)
            action = episode.action(step)
            reward = episode.reward(step)
            xy = state.coords()
            x,y = xy[0],xy[1]

            # find the maximum action value from the next state
            next_state = episode.state(step+1)
            next_xy = next_state.coords()
            n_x,n_y = next_xy[0],next_xy[1]
                
            # find greatest q at next state
            next_q = self._Q[n_x, n_y, :]
            max_next_q = max(next_q)
            
            if step == steps - 2: # terminal step has a terminating value equal to its reward
                max_next_q = episode.reward(step+1)

            # calculate new q
            old_q = self._Q[x,y,action]
            new_q = old_q + self.alpha()*(reward+self.gamma()*max_next_q-old_q)
            
            # update q
            self._update_q_and_policy(xy,action,new_q)
            
        
