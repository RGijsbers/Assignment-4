#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning experiments
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent,PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth

def run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, learning_rate, gamma,
                    epsilon, n_planning_updates):
    repetition = 0
    learning_curve_average = np.zeros(shape=(n_repetitions, n_timesteps)) #use this to help plot the rewards for the learning curve
    env = WindyGridworld() #initialize environment
    s = env.reset() #make sure it starts at the right position

    if policy == 'Dyna': #check which agent needs to be made
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) #initialize Dyna agent
    if policy == 'Prioritized Sweeping':
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) #initialize Prioritized Sweeping agent

    while repetition < n_repetitions: #do the right amount of repetitions
        s = env.reset()
        episode_reward = 0
        for t in range(1, n_timesteps):
            # Select action, transition, update policy
            a = pi.select_action(s,epsilon) #select action
            s_next,r,done = env.step(a) #take step
            pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates) #update policy
            episode_reward += r #update reward counter
            learning_curve_average[repetition][t] = episode_reward/t #update to get the learning curve
            if done:
                s = env.reset() #reset the environment
            else:
                s = s_next #continue in the same environment
        repetition += 1
    
    learning_curve = np.average(learning_curve_average, axis=0) #average the learning curve over all repetitions
    learning_curve = smooth(learning_curve,smoothing_window) #apply additional smoothing
    return learning_curve


def experiment():

    n_timesteps = 10000
    n_repetitions = 10
    smoothing_window = 101
    gamma = 0.99

    for policy in ['Dyna']: #,'Prioritized Sweeping'
    
        ##### Assignment a: effect of epsilon ######
        learning_rate = 0.5
        n_planning_updates = 5
        epsilons = [0.01,0.05,0.1,0.25]
        Plot = LearningCurvePlot(title = '{}: effect of $\epsilon$-greedy'.format(policy))
        
        for epsilon in epsilons:
            learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                           learning_rate, gamma, epsilon, n_planning_updates)
            Plot.add_curve(learning_curve,label='$\epsilon$ = {}'.format(epsilon))
        # Plot.save('pease_why.png'.format(policy))
        Plot.save('{}_egreedy.png'.format(policy))
        
        #### Assignment b: effect of n_planning_updates ######
        epsilon=0.1
        n_planning_updatess = [1,5,15]
        learning_rate = 0.5
        Plot = LearningCurvePlot(title = '{}: effect of number of planning updates per iteration'.format(policy))

        for n_planning_updates in n_planning_updatess:
            learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                           learning_rate, gamma, epsilon, n_planning_updates)
            Plot.add_curve(learning_curve,label='Number of planning updates = {}'.format(n_planning_updates))
        Plot.save('{}_n_planning_updates.png'.format(policy))  
        
        ##### Assignment 1c: effect of learning_rate ######
        epsilon=0.1
        n_planning_updates = 5
        learning_rates = [0.1,0.5,1.0]
        Plot = LearningCurvePlot(title = '{}: effect of learning rate'.format(policy))
    
        for learning_rate in learning_rates:
            learning_curve=run_repetitions(policy, n_repetitions, n_timesteps, smoothing_window, 
                                           learning_rate, gamma, epsilon, n_planning_updates)
            Plot.add_curve(learning_curve,label='Learning rate = {}'.format(learning_rate))
        Plot.save('{}_learning_rate.png'.format(policy)) 
    
if __name__ == '__main__':
    experiment()
