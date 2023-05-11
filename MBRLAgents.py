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
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

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


class PrioritizedSweepingAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, max_queue_size=200, priority_cutoff=0.01):
        self.n_states = n_states #initialize all variables needed
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff
        self.queue = PriorityQueue()
        self.Q_sa = np.zeros((n_states,n_actions))
        self.transition_counts = np.zeros((n_states, n_actions, n_states))
        self.reward_sum = np.zeros((n_states, n_actions, n_states))
        self.simulation = np.zeros((n_states, n_actions, n_states))

    def select_action(self, s, epsilon):
        highest_sa = max(self.Q_sa[s])
        if random.random() > epsilon:
            a = np.where(self.Q_sa[s] == highest_sa)[0][0]
        else:
            a = np.random.randint(0,self.n_actions)

        if self.Q_sa[s,0] == self.Q_sa[s,1] == self.Q_sa[s,2] == self.Q_sa[s,3]:
            a = np.random.randint(0,self.n_actions)
        return a
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        max_Q_sa_next = max(self.Q_sa[s_next])
        p = abs(r + self.gamma * max_Q_sa_next - self.Q_sa[s,a])
        #update priorities in queue
        if p > self.priority_cutoff:
            #Put (s,a) on the queue with priority p (needs a minus since the queue pops the smallest priority first)
            self.queue.put((-p,(s,a)))
        
        self.transition_counts[s,a,s_next] += 1 #update transition counts
        self.reward_sum[s,a,s_next] += r #update reward sum
        self.simulation[s,a,s_next] = self.transition_counts[s,a,s_next]/np.sum(self.transition_counts[s, a, :]) #estimate transition function
        
        for k in range(n_planning_updates):
            if self.queue.empty():
                break
              #Retrieve the top (s,a) from the queue
            _,(s,a) = self.queue.get() # get the top (s,a) for the queue
            #find s_next and r using (s,a) and model
            s_prime = np.random.choice(range(self.n_states), p = self.simulation[s,a])
            r_simulation = self.reward_sum[s,a,s_prime]/self.transition_counts[s,a,s_prime]
            max_Q_sa_prime = max(self.Q_sa[s_prime])
            self.Q_sa[s,a] = self.Q_sa[s,a] + self.learning_rate * (r_simulation + (self.gamma*max_Q_sa_prime) - self.Q_sa[s,a]) # update Q_sa
            
            for pre_s in range(self.n_states):
                for pre_a in range(self.n_actions):
                  reverse_model = self.transition_counts[pre_s,pre_a,s]/np.sum(self.transition_counts[:, :, s])
                  if reverse_model > 0:
                      pre_r = self.reward_sum[pre_s,pre_a,s]/self.transition_counts[pre_s,pre_a,s]
                      p = abs(pre_r + self.gamma * max_Q_sa_next - self.Q_sa[pre_s,pre_a])
                      if p > self.priority_cutoff:
                        #Put (s,a) on the queue with priority p (needs a minus since the queue pops the smallest priority first)
                        self.queue.put((-p,(s,a)))
            
def test():

    n_timesteps = 1000
    gamma = 0.99

    # Algorithm parameters
    policy = 'dyna' # 'ps' 
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
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
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
