#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
import random
import math
import torch
import torch.nn as nn
from A4Environment import WindyGridworld


class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states #initialize all variables needed
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((self.n_states, self.n_actions))
        self.simulation = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.transition_counts = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.reward_sum = np.zeros((self.n_states, self.n_actions, self.n_states))
        
    def select_action(self, s, epsilon):
        highest_sa = max(self.Q_sa[s]) #the best action according to the Q table
        smart_action = np.where(self.Q_sa[s] == highest_sa)[0][0] #get the right action number out of it

        if np.random.rand() > epsilon: #do the smart action
            a = smart_action
        else:
            a = np.random.randint(0,self.n_actions) #do a random action
        return a
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        self.transition_counts[s][a][s_next] += 1 #update transition counts
        self.reward_sum[s][a][s_next] += r #update reward sum
        self.Q_sa[s][a] = self.Q_sa[s][a] + self.learning_rate * (r + (self.gamma*max(self.Q_sa[s_next])) - self.Q_sa[s][a]) # update Q_sa
        self.simulation[s][a][s_next] = self.transition_counts[s][a][s_next]/sum(self.transition_counts[s][a]) #estimate transition function

        for k in range(n_planning_updates): #do things in the model
            not_zero_states = np.ndarray.nonzero(self.transition_counts) #get all states which the agent has been in
            s_already_taken = np.random.choice(not_zero_states[0]) #get a random state already visited by the agent
            not_zero_actions = np.ndarray.nonzero(self.transition_counts[s_already_taken]) #get all the actions already done in that state by the agent
            a_already_taken = np.random.choice(not_zero_actions[0]) #get a random action already done in that sate by the agent
            s_next = np.random.choice(range(self.n_states), p = self.simulation[s_already_taken][a_already_taken]) #get the next state from that action and state already taken

            r_simulation = self.reward_sum[s_already_taken][a_already_taken][s_next]/self.transition_counts[s_already_taken][a_already_taken][s_next] #get the simulation reward
            self.Q_sa[s_already_taken][a_already_taken] = self.Q_sa[s_already_taken][a_already_taken] + self.learning_rate * (r_simulation + (self.gamma*max(self.Q_sa[s_already_taken])) - self.Q_sa[s_already_taken][a_already_taken]) # update Q_sa
            pass
        pass


class A2CActor(nn.Module):
    def __init__(self,  n_states, n_actions, learning_rate):
        super(A2CActor, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)
        self.weights = np.random.randint(-1, 1, size=(n_states, n_actions))
        self.learning_rate = learning_rate

    def select_action(self, s, epsilon):
        highest_sa = max(self.weights[s]) #the best action according to the Q table
        smart_action = np.where(self.weights[s] == highest_sa)[0][0] #get the right action number out of it

        if np.random.rand() > epsilon: #do the smart action
            a = smart_action
        else:
            a = np.random.randint(0,self.n_actions) #do a random action
        return a

    pass

class A2CCritic:
    pass


class A2CAgent:
    pass


class QLearningAgent(object):

    def __init__(self, n_states, n_actions, learning_rate, epsilon, gamma):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((self.n_states, self.n_actions))
    
    def select_action(self, s, epsilon):
        highest_sa = max(self.Q_sa[s]) #the best action according to the Q table
        smart_action = np.where(self.Q_sa[s] == highest_sa)[0][0] #get the right action number out of it

        if np.random.rand() > epsilon: #do the smart action
            a = smart_action
        else:
            a = np.random.randint(0,self.n_actions) #do a random action
        return a

    def update(self,s,a,r,done,s_next,n_planning_updates):
        self.Q_sa[s][a] = self.Q_sa[s][a] + self.learning_rate*(r + self.gamma*(self.gamma*max(self.Q_sa[s_next])) - self.Q_sa[s][a])
        s = s_next
        pass

def test():

    n_timesteps = 1000
    gamma = 0.99

    # Algorithm parameters
    policy =  'Q' # 'dyna' 'A2C' 
    epsilon = 0.1
    learning_rate = 0.5
    n_planning_updates = 5

    # Plotting parameters
    plot = False
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'A2C':    
        pi = A2CActor(env.n_states,env.n_actions,learning_rate,gamma) # Initialize A2C policy
    elif policy == 'Q':
        pi = QLearningAgent(env.n_states,env.n_actions,learning_rate,epsilon,gamma) # Initialize Q-learning policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = True
    
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
            
    
if __name__ == '__main__':
    test()
