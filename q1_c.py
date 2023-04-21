#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

from common.scenarios import test_three_row_scenario
from common.airport_map_drawer import AirportMapDrawer

from td.td_policy_predictor import TDPolicyPredictor
from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

import numpy as np
import matplotlib.pyplot as plt

import random

np.random.seed(42)
random.seed(42)

if __name__ == '__main__':
    airport_map, drawer_height = test_three_row_scenario()
    env = LowLevelEnvironment(airport_map)
    env.set_nominal_direction_probability(0.8)

    # Policy to evaluate
    pi = env.initial_policy()
    pi.set_epsilon(0)
    pi.set_action(14, 1, LowLevelActionType.MOVE_DOWN)
    pi.set_action(14, 2, LowLevelActionType.MOVE_DOWN)  
    
    # Policy evaluation algorithm
    pe = PolicyEvaluator(env)
    pe.set_policy(pi)
    v_pe = ValueFunctionDrawer(pe.value_function(), drawer_height)  
    pe.evaluate()
    v_pe.update()
    # Calling update a second time clears the "just changed" flag
    # which means all the digits will be rendered in black
    v_pe.update()  
    
    # Off policy MC predictors
    num_epsilons = 10
    epsilon_b_values = [(i+1)/num_epsilons for i in range(num_epsilons)]
    
    num_values = len(epsilon_b_values)
    
    mc_predictors = [None] * num_values
    mc_drawers = [None] * num_values

    for i in range(num_values):
        mc_predictors[i] = OffPolicyMCPredictor(env)
        mc_predictors[i].set_use_first_visit(True)
        b = env.initial_policy()
        b.set_epsilon(epsilon_b_values[i])
        mc_predictors[i].set_target_policy(pi)
        mc_predictors[i].set_behaviour_policy(b)
        mc_predictors[i].set_experience_replay_buffer_size(64)
        mc_drawers[i] = ValueFunctionDrawer(mc_predictors[i].value_function(), drawer_height)
        
    episodes = 10
    for e in range(episodes):
        print(f'{e+1} / {episodes}')
        for i in range(num_values):
            mc_predictors[i].evaluate()
            mc_drawers[i].update()
       
    v_pe.save_screenshot("q1_c_truth_pe.pdf")
    for i in range(num_values):
        mc_drawers[i].save_screenshot(f"mc-off-{int(epsilon_b_values[i]*10):03}-pe.pdf")


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
    for i in range(num_values):
        values = list_values(mc_predictors[i],airport_map)
        plot[epsilon_b_values[i]] = values

    width = 1/(2+num_values)
    bar = 0
    for epsilon in plot:
        data = plot[epsilon]
        if isinstance(epsilon,int):
            epsilon = round(epsilon,2)
        ind = np.arange(len(plot[epsilon]))
        ax.bar(ind+width*bar,data,width,label=epsilon)
        bar += 1

    ax.set_ylabel('Value',fontsize = 25)
    ax.set_xlabel('Cell',fontsize=25)
    ax.set_title('Value Functions',fontsize = 30)

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    ax.legend(loc='best',fontsize=30)
    plt.show()
