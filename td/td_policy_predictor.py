'''
Created on 8 Mar 2023

@author: ucacsjj
'''
from monte_carlo.episode_sampler import EpisodeSampler

from .td_algorithm_base import TDAlgorithmBase

class TDPolicyPredictor(TDAlgorithmBase):

    def __init__(self, environment):
        
        TDAlgorithmBase.__init__(self, environment)
        
        self._minibatch_buffer= [None]
                
    def set_target_policy(self, policy):        
        self._pi = policy        
        self.initialize()
        self._v.set_name("TDPolicyPredictor")
        
    def evaluate(self):
        
        episode_sampler = EpisodeSampler(self._environment)
        
        for episode in range(self._number_of_episodes):

            # Choose the start for the episode            
            start_x, start_a  = self._select_episode_start()
            self._environment.reset(start_x) 
            
            # Now sample it
            new_episode = episode_sampler.sample_episode(self._pi, start_x, start_a)

            # If we didn't terminate, skip this episode
            if new_episode.terminated_successfully() is False:
                continue
            
            # Update with the current episode
            self._update_value_function_from_episode(new_episode)
            
            # Pick several randomly from the experience replay buffer and update with those as well
            for _ in range(min(self._replays_per_update, self._stored_experiences)):
                episode = self._draw_random_episode_from_experience_replay_buffer()
                self._update_value_function_from_episode(episode)
                
            self._add_episode_to_experience_replay_buffer(new_episode)
            
    def _update_value_function_from_episode(self, episode):

        steps = episode.number_of_steps()
        for step in range(steps-1):
            # extract information from step
            state = episode.state(step)
            reward = episode.reward(step)
            xy = state.coords()
            x,y = xy[0],xy[1]

            # find the value of the next step
            next_state = episode.state(step+1)
            xy_next = next_state.coords()
            next_v = self._v.value(xy_next[0],xy_next[1])

            # calculate new value function at state
            old_v = self._v.value(x,y)
            new_v = old_v + self.alpha()*(reward+self.gamma()*next_v - old_v)

            # update state value function
            self._v.set_value(x,y,new_v)
            
            


