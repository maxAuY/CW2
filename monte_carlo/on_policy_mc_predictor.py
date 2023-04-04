'''
Created on 20 Feb 2023

@author: ucacsjj
'''

import numpy as np

from .monte_carlo_policy_predictor import MonteCarloPolicyPredictor

class OnPolicyMCPredictor(MonteCarloPolicyPredictor):
    '''
    classdocs
    '''

    def __init__(self, environment):
        
        MonteCarloPolicyPredictor.__init__(self, environment)            

    def initialize(self):
        
        MonteCarloPolicyPredictor.initialize(self)
                
        # Grids for computing values
        # Get the environment and environment_map
        environment = self._environment
        environment_map = environment.environment_map()
        w = environment_map.width()
        h = environment_map.height()
        self._returns_grid = np.zeros((w, h))
        self._count_grid = np.zeros((w, h))
            
        self._v.set_name("OnPolicyMCPredictor")
            
    # For the on policy, the start action has to be according to the 
    # policy and can't be random.
    def _select_episode_start(self):
        
        if self._use_exploring_starts is True:
            start_x = self._environment.random_initial_state()
        else:
            start_x = self._environment.get_state(0, 0)
        
        coords = start_x.coords()
        start_a = self._pi.action(coords[0], coords[1])
        #start_a = self._environment.random_initial_action(start_x)
        
        return start_x, start_a
            
    def _update_value_function_from_episode(self, episode):


        # Very simple way to implement first visit
        environment_map = self._environment.environment_map()
        w = environment_map.width()
        h = environment_map.height()
        visited_grid = np.zeros((w, h))

        G = 0

        for s in range(episode.number_of_steps() - 1, -1, -1):
            
            # Work out the return              
            G = self._gamma * G + episode.reward(s)
            
            coords = episode.state(s).coords()
            
            if (self._use_first_visit is True) and (visited_grid[coords[0], coords[1]] > 0):
                continue
            
            self._returns_grid[coords[0], coords[1]] += G
            self._count_grid[coords[0], coords[1]] += 1
            visited_grid[coords[0], coords[1]] = 1
            
            average_return = self._returns_grid[coords[0], coords[1]] / self._count_grid[coords[0], coords[1]]
            
            self._v.set_value(coords[0], coords[1], average_return)

            
            
            
        