import gym
from gym import spaces

import numpy as np
import config
from config import debug

from colorama import Fore, Back, Style

REWARD_PENALTY = config.REWARD_PENALTY



def prepare_action_dicts(df):

    def get_col_dict(colname):
        l = np.sort(df[colname].unique())
        l_p1 = np.roll(l, shift=-1)
        l_p1[-1] = -1 #invalid choice
        l_m1 = np.roll(l, shift=1)
        l_m1[0] = -1 #invalid choice
        d = {}
        for idx, elem in enumerate(l):
            d[elem] = {-1: l_m1[idx], 0: elem, 1: l_p1[idx]}
        return d

    d = {}
    col_list = []
    for colname in ['itr', 'dvfs']:
        col_list.append(colname)
        d[colname] = get_col_dict(colname)

    return d, col_list




class WorkloadEnv(gym.Env):
    def __init__(self, df):
        super(WorkloadEnv, self).__init__()

        misc_cols = ['joules_99', 'joules_per_interrupt', 'time_per_interrupt']

        if 0:
            state_dict = df[(df['exp']==0) & (df['core']==0)].drop('fname', axis=1).set_index(['itr', 'dvfs', 'rapl', 'sys', 'core', 'exp']).T.to_dict()
            df_state = df[(df['exp']==0) & (df['core']==0)].drop('fname', axis=1).set_index(['itr', 'dvfs', 'rapl', 'sys', 'core', 'exp']).drop(misc_cols, axis=1)

        df_state = df.set_index(['itr', 'dvfs', 'qps']).drop(misc_cols, axis=1)
        df_misc = df.set_index(['itr', 'dvfs', 'qps'])[misc_cols]
        state_dict = df_state.T.to_dict()
        reward_dict = df_misc.T.to_dict()

        if debug:
                print('------------------------------------------------')
                print(Fore.BLACK + Back.GREEN + "df_state: " + Style.RESET_ALL)
                print(df_state)
                print('------------------------------------------------')
                print(Fore.BLACK + Back.GREEN + "df_misc: " + Style.RESET_ALL)
                print(df_misc)

        self.state_dict = state_dict
        self.reward_dict = reward_dict
        # key_list: list of all (itr, dvfs, qps) tuples
        self.key_list = list(self.state_dict.keys())
        # action_space: dictionary of all possible actions to take on itr and dvfs 
        self.action_space, self.col_list = prepare_action_dicts(df)
	# state, reward, key: state and reward for key at idx = 4 
        self.state, self.reward, self.key = self.init_state()

        if debug:
                print('------------------------------------------------')
                print(Fore.BLACK + Back.GREEN + "action_space: " + Style.RESET_ALL)
                print(self.action_space)
                print(Fore.BLACK + Back.GREEN + "col_list: " + Style.RESET_ALL)
                print(self.col_list)
                print('------------------------------------------------')
                print(Fore.BLACK + Back.GREEN + "init state, init reward, init key: " + Style.RESET_ALL)
                print(self.state)
                print(self.reward)
                print(self.key)

        #limit should be in n_requests
        self.time_limit_sec = 30
        self.current_time_sec = 0



    def step(self, action, debug=False):
        print('------------------------------------------------')
        print("ENV STEP..")

        # action will be itr delta (-1, 0, +1), dvfs delta (-1, 0, +1)
        new_key = list(self.key)

        for idx in range(len(action)-1):            
            col = self.col_list[idx]
            new_key[idx] = self.action_space[col][self.key[idx]][action[idx]]
            if not debug:
                print(f'{col} {self.key[idx]} action={action[idx]} {new_key[idx]}')
        
        new_key = tuple(new_key)

        if not debug:
            print("----------------")
            print("Old config: ", self.key)
            print("New config: ", new_key)

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

        print('------------------------------------------------')
        print(Fore.BLACK + Back.GREEN + "State IDX = " + str(idx) + Style.RESET_ALL)
        print(Fore.BLACK + Back.GREEN + "State Vector: " + Style.RESET_ALL)
        print(state)
        print(Fore.BLACK + Back.GREEN + "Reward Vector: " + Style.RESET_ALL)
        print(reward)

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



