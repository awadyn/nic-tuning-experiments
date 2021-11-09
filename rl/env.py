import gym
from gym import spaces

import numpy as np

class Workload(gym.Env):
    def __init__(self, df):
        super(Workload, self).__init__()

        state_dict = df[(df['exp']==0) & (df['core']==0)].drop('fname', axis=1).set_index(['itr', 'dvfs', 'rapl', 'sys', 'core', 'exp']).T.to_dict()

        self.state_dict = state_dict #maps (itr, dvfs, rapl, [qps, workload]) -> state
        self.key_list = list(self.state_dict.keys())

        self.state, self.key = self.init_state()

        self.time_limit_sec = 30
        self.current_time_sec = 0

    def step(self, action, debug=False):
        #action will be itr delta, rapl delta, dvfs delta
        new_key = list(self.key)
        for idx in range(len(action)):
            new_key[idx] += action[idx]
        new_key = tuple(new_key)

        if debug:
            print(self.key)
            print(new_key)

        if new_key not in self.key_list:
            raise NotImplementedError() #open question: terminate the experiment

        #compute reward from previous state
        reward = self.state['joules_per_interrupt'] #TODO: multiply latency here

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

        key = self.key_list[idx]

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