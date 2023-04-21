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

import random
import matplotlib.pyplot as plt

if __name__ == '__main__':
    np.random.seed(52)
    random.seed(52)

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
    policy_learner._replays_per_update = 0
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
    iterations = 40
    avg_ep_lengths = []

    for i in range(iterations):
        print(i)
        pi.set_epsilon(1/math.sqrt(1+0.25*i))

        start_time = time.time() # start timing
        avg_ep_length,avg_ep_reward = policy_learner.find_policy()
        end_time = time.time() # stop timing

        value_function_drawer.update()
        greedy_optimal_policy_drawer.update()
        
        # print(f"epsilon={1/math.sqrt(1+i)};alpha={policy_learner.alpha()}")

        # track times and epsilons
        epsilon = pi._epsilon
        epsilons.append(epsilon)
        times.append(end_time-start_time)
        avg_ep_lengths.append(avg_ep_length)

    table = np.zeros((iterations,3))
    table[:,0] = epsilons
    table[:,1] = times
    table[:,2] = avg_ep_lengths

    print(table)
    
    i = 0
    d = 4
    x = epsilons[i:]
    y = times[i:]
    for degree in range(1,d+1):
        plt.scatter(x,y)
        plt.title('Time vs Epsilon',fontsize=30)
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, degree))(np.unique(x)))
        plt.xlabel('Epsilon',fontsize=30)
        plt.ylabel('Time [s]',fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()

    x = avg_ep_lengths[i:]
    y = times[i:]
    for degree in range(1,d+1):
        plt.scatter(x,y)
        plt.title('Time vs average episode length',fontsize=30)
        plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, degree))(np.unique(x)))
        plt.xlabel('Average episode length',fontsize=30)
        plt.ylabel('Time [s]',fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.show()
        