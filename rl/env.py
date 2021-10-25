import gym
from gym import spaces

import numpy as np

class Workload(gym.Env):
    def __init__(self, state_dict):
        super(Workload, self).__init__()

        self.state_dict = state_dict 

        self.state = self.init_state()

    def step(self, action):
        

    def reset(self):
        self.state = self.init_state

    def render(self):
        pass

    def init_state(self):
        idx = np.random.randint(len(self.state_dict))

        self.state = self.state_dict[list(self.state_dict.keys())[idx]]

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