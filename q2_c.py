#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math

from common.scenarios import corridor_scenario

from common.airport_map_drawer import AirportMapDrawer


from td.q_learner import QLearner

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

import matplotlib.pyplot as plt
import numpy as np
import random

if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
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
    
    for i in range(40):
        print(i)
        policy_learner.find_policy()
        value_function_drawer.update()
        greedy_optimal_policy_drawer.update()
        pi.set_epsilon(1/math.sqrt(1+0.25*i))
        print(f"epsilon={1/math.sqrt(1+0.25*i)};alpha={policy_learner.alpha()}")

    value_function_drawer.save_screenshot('q_learning_values.pdf')
    greedy_optimal_policy_drawer.save_screenshot('q_learning_policy.pdf')
        
    # Policy evaluation algorithm
    pe = PolicyEvaluator(env)
    pe.set_policy(policy_learner.policy())
    v_pe = ValueFunctionDrawer(pe.value_function(), drawer_height)  
    pe.evaluate()
    v_pe.update()

    v_pe.save_screenshot('q_learning_values_truth.pdf')

    def list_values(predictor,airport_map):
        all_values = []
        for y in range(airport_map._height):
                for x in range(airport_map._width):
                    value = predictor._v._values[x,y]
                    if not np.isnan(value): 
                        if not value == 0:
                            all_values.append(value)   
        return all_values
    
    fig,ax = plt.subplots()
    plot = {}
    plot['truth'] = list_values(pe,airport_map)
    plot['Q-learning'] = list_values(policy_learner,airport_map)

    width = 1/3    
    bar = 0
    for learner in plot:
        data = plot[learner]
        ind = np.arange(len(plot[learner]))
        ax.bar(ind+width*bar,data,width,label=learner)
        bar += 1

    ax.set_ylabel('Value',fontsize=25)
    ax.set_xlabel('Cell',fontsize=25)
    ax.set_title('Value Functions',fontsize=30)

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    ax.legend(loc='best',fontsize=30)
    plt.show()