#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math
import time
import numpy as np

from common.scenarios import corridor_scenario

from common.airport_map_drawer import AirportMapDrawer


from td.q_learner import QLearner

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

if __name__ == '__main__':
    airport_map, drawer_height = corridor_scenario()

    # Show the scenario        
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()
    
    # Create the environment
    env = LowLevelEnvironment(airport_map)

    # Extract the initial policy. This is e-greedy
    pi = env.initial_policy()
    
    # Select the controller
    policy_learner = QLearner(env)   
    policy_learner.set_initial_policy(pi)

    # These values worked okay for me.
    policy_learner.set_alpha(0.1)
    policy_learner.set_experience_replay_buffer_size(64)
    policy_learner.set_number_of_episodes(32)
    
    # The drawers for the state value and the policy
    value_function_drawer = ValueFunctionDrawer(policy_learner.value_function(), drawer_height)    
    greedy_optimal_policy_drawer = LowLevelPolicyDrawer(policy_learner.policy(), drawer_height)
    
    times = []
    epsilons = []
    episodes = 40
    for i in range(episodes):
        print(i)

        start_time = time.time() # start timing
        policy_learner.find_policy()
        end_time = time.time() # stop timing

        value_function_drawer.update()
        greedy_optimal_policy_drawer.update()
        pi.set_epsilon(1/math.sqrt(1+0.25*i))
        print(f"epsilon={1/math.sqrt(1+i)};alpha={policy_learner.alpha()}")

        # track times and epsilons
        epsilons.append(1/math.sqrt(1+i))
        times.append(end_time-start_time)

    table = np.zeros((episodes,2))
    table[:,0] = epsilons
    table[:,1] = times

    print(table)
    
        