#!/usr/bin/env python3

'''
Created on 7 Mar 2023

@author: steam
'''

from common.scenarios import test_three_row_scenario
from common.airport_map_drawer import AirportMapDrawer

from monte_carlo.on_policy_mc_predictor import OnPolicyMCPredictor
from monte_carlo.off_policy_mc_predictor import OffPolicyMCPredictor

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer
from generalized_policy_iteration.policy_evaluator import PolicyEvaluator

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.ticker import ScalarFormatter

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
    v_pe.update()  
    
    run_variance = False
    if run_variance:
        # On policy MC predictor
        mcpp = OnPolicyMCPredictor(env)
        mcpp.set_target_policy(pi)
        mcpp.set_experience_replay_buffer_size(64)
        
        # Q1b: Experiment with this value
        first_visit = True
        mcpp.set_use_first_visit(first_visit)
        
        v_mcpp = ValueFunctionDrawer(mcpp.value_function(), drawer_height)
        
        # Off policy MC predictor
        mcop = OffPolicyMCPredictor(env)
        mcop.set_target_policy(pi)
        mcop.set_experience_replay_buffer_size(64)
        b = env.initial_policy()
        b.set_epsilon(0.2)
        mcop.set_behaviour_policy(b)
        
        # Q1b: Experiment with this value
        mcop.set_use_first_visit(first_visit)

        v_mcop = ValueFunctionDrawer(mcop.value_function(), drawer_height)
            
        # iterating to find variance
        def get_final_variance(predictor):
            all_values = []
            for x in range(14):
                    for y in range(3):
                        value = predictor._v._values[x,y]
                        if not np.isnan(value): 
                            if not value == 0:
                                all_values.append(value)   
            variance = np.var(all_values)
            return variance
            
        iterations = 2
        no_episodes = 100
        on_variances = []
        off_variances = []
        for iteration in range(iterations):
            print(f'{iteration+1} / {iterations}')
            # On policy MC predictor
            mcpp = OnPolicyMCPredictor(env)
            mcpp.set_target_policy(pi)
            mcpp.set_experience_replay_buffer_size(64)
            
            # Q1b: Experiment with this value
            mcpp.set_use_first_visit(first_visit)
            
            # Off policy MC predictor
            mcop = OffPolicyMCPredictor(env)
            mcop.set_target_policy(pi)
            mcop.set_experience_replay_buffer_size(64)
            b = env.initial_policy()
            b.set_epsilon(0.2)
            mcop.set_behaviour_policy(b)
            
            # Q1b: Experiment with this value
            mcop.set_use_first_visit(first_visit)

            onp_times = []
            offp_times = []
            for e in range(no_episodes):
                st_op = time.time()
                mcpp.evaluate()
                ed_op = time.time()
                v_mcpp.update()
                st_ofp = time.time()
                mcop.evaluate()
                ed_ofp = time.time()
                v_mcop.update()
                onp_times.append(ed_op-st_op)
                offp_times.append(ed_ofp-st_ofp)

            variance = get_final_variance(mcpp)
            on_variances.append(variance)

            variance = get_final_variance(mcop)
            off_variances.append(variance)

        avg_time_op = np.average(onp_times)
        avg_time_off_policy = np.average(offp_times)
        avg_variance_on = np.average(on_variances)
        avg_variance_off = np.average(off_variances)
        print()
        print('first visit:',first_visit)
        print('avg on variance',avg_variance_on)
        print('avg off variance',avg_variance_off)
        print('avg_time on policy',avg_time_op*no_episodes,'s')
        print('avg_time off policy',avg_time_off_policy*no_episodes,'s')
        print()

        # Sample way to generate outputs    
        v_pe.save_screenshot("q1_b_truth_pe.pdf")
        v_mcop.save_screenshot("q1_b_mc-off_pe.pdf")
        v_mcpp.save_screenshot("q1_b_mc-on_pe.pdf")


    # create graphs for on or off policy
    def list_values(predictor):
        all_values = []
        for y in range(3):
                for x in range(15):
                    value = predictor._v._values[x,y]
                    if not np.isnan(value): 
                        if not value == 0:
                            all_values.append(value)   
        return all_values

    on_policy = False # set on or off policy graphing
    no_episodes = 100
    first_visit = [True,False]
    policy_predictors = [None,None]
    for i in range(2):
        visit = first_visit[i]

        if on_policy:
            # On policy MC predictor
            policy_predictors[i] = OnPolicyMCPredictor(env)
            policy_predictors[i].set_target_policy(pi)
            policy_predictors[i].set_experience_replay_buffer_size(64)
            
            # Q1b: Experiment with this value
            policy_predictors[i].set_use_first_visit(visit)
        else:
            # Off policy MC predictor
            policy_predictors[i] = OffPolicyMCPredictor(env)
            policy_predictors[i].set_target_policy(pi)
            policy_predictors[i].set_experience_replay_buffer_size(64)
            b = env.initial_policy()
            b.set_epsilon(0.2)
            policy_predictors[i].set_behaviour_policy(b)

        for e in range(no_episodes):
            policy_predictors[i].evaluate()
            policy_predictors[i].evaluate()

    fig,ax = plt.subplots()
    plot = {}
    plot['truth'] = list_values(pe)
    for i in range(2):
        values = list_values(policy_predictors[i])
        plot[first_visit[i]] = values

    width = 0.25
    ind = np.arange(len(plot['truth']))
    bar = 0
    for visit in plot:
        data = plot[visit]
        if visit:
            label = 'first visit'
        else:
            label = 'every visit'
        if visit == 'truth':
            label = visit
        ax.bar(ind+width*bar,data,width,label=label)
        bar += 1

    ax.set_ylabel('Value',fontsize=25)
    ax.set_xlabel('Cell',fontsize=25)

    if on_policy:
        title = 'On Policy'
    else:
        title = 'Off Policy'
    ax.set_title(f'{title} Value Functions',fontsize = 30)

    plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=30)    # fontsize of the tick labels
    plt.rc('legend', fontsize=30)    # legend fontsize

    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    ax.legend(loc='best')
    y_formatter = ScalarFormatter(useOffset=True)
    ax.yaxis.set_major_formatter(y_formatter)
    plt.show()