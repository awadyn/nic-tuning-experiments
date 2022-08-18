import gym
from gym import spaces

import numpy as np
import config

REWARD_PENALTY = config.REWARD_PENALTY

'''
reward -> add latency requirement
remove read_lat from features
toggle between two actions and three actions
reward penalty toggle
'''

def prepare_action_dicts(df):

    def get_col_dict(colname):
        l = np.sort(df[colname].unique())
        #print(l)
        l_p1 = np.roll(l, shift=-1)
        #print(l_p1)
        l_p1[-1] = -1 #invalid choice
        
        l_m1 = np.roll(l, shift=1)
        #print(l_m1)
        l_m1[0] = -1 #invalid
        
        d = {}
        for idx, elem in enumerate(l):
            d[elem] = {-1: l_m1[idx], 0: elem, 1: l_p1[idx]}

        return d

    d = {}
    col_list = []
    #for colname in ['itr', 'dvfs', 'rapl']:
    for colname in ['itr', 'dvfs']:
        col_list.append(colname)
        d[colname] = get_col_dict(colname)

    return d, col_list


class WorkloadEnv(gym.Env):
    def __init__(self, df):
        super(WorkloadEnv, self).__init__()

        #basic data structures
        #state_dict = df[(df['exp']==0) & (df['core']==0)].drop('fname', axis=1).set_index(['itr', 'dvfs', 'rapl', 'sys', 'core', 'exp']).T.to_dict()

        misc_cols = ['joules_99']

#        df_state = df[(df['exp']==0) & (df['core']==0)].drop('fname', axis=1).set_index(['itr', 'dvfs', 'rapl', 'sys', 'core', 'exp']).drop(misc_cols, axis=1)

        df_state = df.set_index(['itr', 'dvfs']).drop(misc_cols, axis=1)
        df_misc = df.set_index(['itr', 'dvfs'])[misc_cols]

        state_dict = df_state.T.to_dict()
        print(df_state)
        print()
        print()

        reward_dict = df_misc.T.to_dict()
        print(df_misc)
        print()
        print()


        self.state_dict = state_dict #maps (itr, dvfs) -> state
        self.reward_dict = reward_dict

        self.key_list = list(self.state_dict.keys())
        self.action_space, self.col_list = prepare_action_dicts(df)
        print(self.action_space)
        print(self.col_list)

        self.state, self.reward, self.key = self.init_state()
        print(self.state)
        print(self.reward)
        print(self.key)

        #limit should be in n_requests
        self.time_limit_sec = 30
        self.current_time_sec = 0

    def step(self, action, debug=False):
        #action will be itr delta, rapl delta, dvfs delta
        new_key = list(self.key)
        for idx in range(len(action)-1):
            
            col = self.col_list[idx]

            new_key[idx] = self.action_space[col][self.key[idx]][action[idx]]
            if debug:
                print(f'{col} {self.key[idx]} action={action[idx]} {new_key[idx]}')
        
        new_key = tuple(new_key)

        if debug:
            print(self.key)
            print(new_key)

        #compute reward from previous state
        done = False
        info = {}
        
        #SANITY CHECK: get to lowest dvfs
        current_reward = 1 * (self.reward['joules_per_interrupt'] / self.reward['time_per_interrupt']) #* self.state['read_99th'] #TODO: multiply latency here

        self.current_time_sec += 1 #self.state['time_per_interrupt'] #replace this by small snapshots from features

        if self.current_time_sec >= self.time_limit_sec:
            done = True

        if new_key not in self.key_list:
            #done = True
            #stay at current state, impose penalty on reward
            new_key = self.key
            current_reward *= REWARD_PENALTY

            #raise NotImplementedError() #open question: terminate the experiment

        #update state
        if not done:
            self.key = new_key
            self.state = self.state_dict[self.key]
            self.reward = self.reward_dict[self.key]

        return self.state, current_reward, done, info

    def reset(self):
        self.state, self.reward, self.key = self.init_state()

        self.current_time_sec = 0

        return self.state

    def render(self):
        pass

    def init_state(self):
        #idx = np.random.randint(len(self.state_dict))
        idx = 4

        key = self.key_list[idx]

        state = self.state_dict[key]

        reward = self.reward_dict[key]

        return state, reward, key

#    def episode_trial(self):
#        self.reset()
#
#        history = [self.key]
#        done = False
#        while not done:
#            action = (np.random.randint(3)-1, np.random.randint(3)-1, 0)
#            print('---------')
#            print(action)
#            _,_,done,_ = self.step(action, debug=True)
#            history.append(self.key)
# 
#        return history

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
