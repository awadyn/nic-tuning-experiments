import gym
from gym import spaces

import numpy as np

class Workload(gym.Env):
    def __init__(self, state_dict):
        super(Workload, self).__init__()

        self.state_dict = state_dict 

        self.state, self.key = self.init_state()

        self.time_limit_sec = 30
        self.current_time_sec = 0

    def step(self, action):
        #action will be itr delta, rapl delta, dvfs delta
        itr_delta, dvfs_delta, rapl_delta = action

        new_key = (self.key[0] + itr_delta, self.key[1] + dvfs_delta, self.key[2] + rapl_delta)

        if new_key not in self.state:
            raise NotImplementedError()

        #compute reward from previous state
        reward = self.state['joules_per_interrupt']

        #update state
        self.key = new_key
        self.state = self.state_dict[self.key]

        self.current_time_sec += self.state['time_per_interrupt']

        done = False
        if self.current_time_sec >= self.time_limit_sec:
            done = True

        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.state, self.key = self.init_state()

    def render(self):
        pass

    def init_state(self):
        idx = np.random.randint(len(self.state_dict))

        key = list(self.state_dict).keys()[idx]

        state = self.state_dict[key]

        return state, key

'''
workload fixed:
    init:
        start with some itr, rapl, dvfs -> read corresponding state
    
    step:
        compute new config: action = (itr, rapl, dvfs) absolute or relative (stochastic)
        lookup new state: state -> look up new state (deterministic)
        return reward:
            read store for per-interrupt energy, time and compute metric based on config
            -> in featurizer, compute per interrupt energy, time




'''