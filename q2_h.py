#!/usr/bin/env python3

'''
Created on 9 Mar 2023

@author: ucacsjj
'''

import math

from common.scenarios import corridor_scenario
from common.airport_map_drawer import AirportMapDrawer


from td.sarsa import SARSA
from td.q_learner import QLearner

from generalized_policy_iteration.value_function_drawer import ValueFunctionDrawer

from p1.low_level_environment import LowLevelEnvironment
from p1.low_level_actions import LowLevelActionType
from p1.low_level_policy_drawer import LowLevelPolicyDrawer

import random
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    random.seed(42)
    np.random.seed(42)
    airport_map, drawer_height = corridor_scenario()

    # Show the scenario        
    airport_map_drawer = AirportMapDrawer(airport_map, drawer_height)
    airport_map_drawer.update()
    
    # Create the environment
    env = LowLevelEnvironment(airport_map)
    env.set_nominal_direction_probability(1)

    bar_graph = False
    if bar_graph:
        # Specify array of learners, renderers and policies
        learners = [None] * 2
        v_renderers = [None] * 2
        p_renderers = [None] * 2 
        pi = [None] * 2
        
        pi[0] = env.initial_policy()
        pi[0].set_epsilon(1)
        learners[0] = SARSA(env)
        learners[0].set_alpha(0.1)
        learners[0].set_experience_replay_buffer_size(64)
        learners[0].set_number_of_episodes(32)
        learners[0].set_initial_policy(pi[0])
        v_renderers[0] = ValueFunctionDrawer(learners[0].value_function(), drawer_height)    
        p_renderers[0] = LowLevelPolicyDrawer(learners[0].policy(), drawer_height)
        
        pi[1] = env.initial_policy()
        pi[1].set_epsilon(1)
        learners[1] = QLearner(env)
        learners[1].set_alpha(0.1)
        learners[1].set_experience_replay_buffer_size(64)
        learners[1].set_number_of_episodes(32)
        learners[1].set_initial_policy(pi[1])      
        v_renderers[1] = ValueFunctionDrawer(learners[1].value_function(), drawer_height)    
        p_renderers[1] = LowLevelPolicyDrawer(learners[1].policy(), drawer_height)

        for i in range(100):
            print(i)
            for l in range(2):
                learners[l].find_policy()
                v_renderers[l].update()
                p_renderers[l].update()
                pi[l].set_epsilon(1/math.sqrt(1+0.25*i))

                if l == 0:
                    name = 'sarsa'
                else:
                    name = 'q-learner'
                iteration = i+1

                # if (not iteration % 50) or (iteration in [10,20,30,40,60,70]):
                    # v_renderers[l].save_screenshot(f'{name}_value_function_{iteration}_iterations.pdf')
                    # p_renderers[l].save_screenshot(f'{name}_policy_{iteration}_iterations.pdf')

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
        learner_names = ['SARSA','Q-learning']
        plot = {}
        for i in range(2):
            values = list_values(learners[i],airport_map)
            plot[learner_names[i]] = list_values(learners[i],airport_map)

        width = 1/3
        bar = 0
        for learner in plot:
            data = plot[learner]
            ind = np.arange(len(plot[learner]))
            ax.bar(ind+width*bar,data,width,label=learner)
            bar += 1

        ax.set_ylabel('Value',fontsize = 25)
        ax.set_xlabel('Cell',fontsize=25)
        ax.set_title('Value Functions',fontsize = 30)

        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        ax.legend(loc='best',fontsize=30)
        plt.show()

    learning_rate = True
    if learning_rate:
        avg_rewards = {}
        learner_names = ['SARSA','Q-learner']

        for learner_name in learner_names:
            avg_rewards[learner_name] = []
            test_alphas = [i*0.1 for i in range(1,11)]
            # test_alphas = [0.1,0.2]

        for alpha in test_alphas:
            # Specify array of learners, renderers and policies
            learners = [None] * 2
            v_renderers = [None] * 2
            p_renderers = [None] * 2 
            pi = [None] * 2
            
            pi[0] = env.initial_policy()
            pi[0].set_epsilon(1)
            learners[0] = SARSA(env)
            learners[0].set_alpha(alpha)
            learners[0].set_experience_replay_buffer_size(5)
            learners[0].set_number_of_episodes(32)
            learners[0].set_initial_policy(pi[0])
            # v_renderers[0] = ValueFunctionDrawer(learners[0].value_function(), drawer_height)    
            # p_renderers[0] = LowLevelPolicyDrawer(learners[0].policy(), drawer_height)
            
            
            pi[1] = env.initial_policy()
            pi[1].set_epsilon(1)
            learners[1] = QLearner(env)
            learners[1].set_alpha(alpha)
            learners[1].set_experience_replay_buffer_size(5)
            learners[1].set_number_of_episodes(32)
            learners[1].set_initial_policy(pi[1])      
            # v_renderers[1] = ValueFunctionDrawer(learners[1].value_function(), drawer_height)    
            # p_renderers[1] = LowLevelPolicyDrawer(learners[1].policy(), drawer_height)

            iterations = 100
            for i in range(iterations):
                print(f'alpha = {alpha}, {i+1} / {iterations}')
                for l in range(2):
                    avg_ep_length,avg_ep_reward = learners[l].find_policy()
                    
                    pi[l].set_epsilon(1/math.sqrt(1+0.25*i))
                    if i == iterations-1:
                        # v_renderers[l].update()
                        # p_renderers[l].update()
                        avg_rewards[learner_names[l]].append(avg_ep_reward)
            
        fig,ax = plt.subplots()
        for learner in learner_names:
            ax.plot(test_alphas,avg_rewards[learner],label=learner)
            ax.set_title('Q-learner vs SARSA performance',fontsize=30)
            ax.set_xlabel('Learning rate',fontsize=25)
            ax.set_ylabel('Average sum of rewards per episode in final iteration',fontsize=25)

        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        ax.legend(loc='best',fontsize=30)
        plt.show()

                    


            