import gym
from gym import spaces

import numpy as np

def prepare_action_dicts(df):

    def get_col_dict(colname):
        l = np.sort(df[colname].unique())
        
        l_p1 = np.roll(l, shift=-1)
        l_p1[-1] = -1 #invalid choice
        
        l_m1 = np.roll(l, shift=1)
        l_m1[0] = -1 #invalid
        
        d = {}
        for idx, elem in enumerate(l):
            d[elem] = {-1: l_m1[idx], 0: elem, 1: l_p1[idx]}

        return d

    d = {}
    col_list = []
    for colname in ['itr', 'dvfs', 'rapl']:
        col_list.append(colname)
        d[colname] = get_col_dict(colname)

    return d, col_list


class Workload(gym.Env):
    def __init__(self, df):
        super(Workload, self).__init__()

        #basic data structures
        state_dict = df[(df['exp']==0) & (df['core']==0)].drop('fname', axis=1).set_index(['itr', 'dvfs', 'rapl', 'sys', 'core', 'exp']).T.to_dict()

        self.state_dict = state_dict #maps (itr, dvfs, rapl, [qps, workload]) -> state
        self.key_list = list(self.state_dict.keys())
        self.action_space, self.col_list = prepare_action_dicts(df)

        self.state, self.key = self.init_state()

        self.time_limit_sec = 30
        self.current_time_sec = 0

    def step(self, action, debug=False):
        #action will be itr delta, rapl delta, dvfs delta
        new_key = list(self.key)
        for idx in range(len(action)):
            
            col = self.col_list[idx]

            new_key[idx] = self.action_space[col][self.key[idx]][action[idx]]
            if debug:
                print(f'{col} {self.key[idx]} action={action[idx]} {new_key[idx]}')
        
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

        self.current_time_sec = 0

    def render(self):
        pass

    def init_state(self):
        idx = np.random.randint(len(self.state_dict))

        key = self.key_list[idx]

        state = self.state_dict[key]

        return state, key

    def episode_trial(self, N=10):
        self.reset()

        for _ in range(N):
            action = (np.random.randint(3)-1, np.random.randint(3)-1, 0)
            print('---------')
            print(action)
            _ = self.step(action, debug=True)


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