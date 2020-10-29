import pandas as pd
import numpy as np
import matplotlib.pylab as plt

import torch
import torch.nn as nn
import torch.optim as optim

from multiprocessing import Manager, Process
from scipy.stats import entropy
import glob

plt.ion()

JOULE_CONVERSION = 0.00001526
TIME_CONVERSION = 1./(2899999*1000)
KEYS = ['MSG', 'ITR', 'RAPL', 'DVFS']
METRIC = 'EDP'

'''
Varying workloads
More parameters
Finer-grained parameters
Other techniques
'''

def find_neighbor_keys(key, int_to_vals, vals_to_int):
    neighbors = []
    for idx, val in enumerate(key):
        if KEYS[idx]=='MSG': #don't look for neighbor in message space
            continue
        integer_val = vals_to_int[KEYS[idx]][val]

        integer_neigh1, integer_neigh2 = integer_val-1, integer_val+1

        #print(integer_val, integer_neigh1, integer_neigh2)
        for i in [integer_neigh1, integer_neigh2]:
            if i in int_to_vals[KEYS[idx]]: #neighbor key should not be out-of-bounds
                new_key = list(key)
                new_key[idx] = int_to_vals[KEYS[idx]][i]

                neighbors.append(tuple(new_key))

    return neighbors

def get_mappers(df):
    int_to_vals, vals_to_int = {}, {}
    for k in KEYS:
        values = df[k].unique()
        int_to_vals[k] = dict(zip(np.arange(len(values)), values))
        vals_to_int[k] = dict(zip(values, np.arange(len(values))))

    return int_to_vals, vals_to_int

def fill_overflows(df):
    d = df[(df['PK0_JOULE'] < 0) | ((df['PK1_JOULE'] < 0))].copy()

    return df.copy()

def check_neighbors(df):
    pass

def prepare_data(filename, msg_size=None):
    df = pd.read_csv(filename, sep = ' ')

    df = fill_overflows(df)

    df['Time'] = (df['END_RDTSC'] - df['START_RDTSC']) * TIME_CONVERSION

    df['Energy'] = df['PK0_JOULE'] + df['PK1_JOULE']

    df['EDP'] = 0.5 * df['Energy'] * df['Time']

    df['Power'] = df['Energy'] / df['Time']

    df['Power_P0'] = df['PK0_JOULE'] / df['Time']

    df['Power_P1'] = df['PK1_JOULE'] / df['Time']

    df['DVFS'] = df['DVFS'].apply(lambda x: int(x, base=16))

    if msg_size is not None: 
        df = df[df['MSG']==msg_size].copy()

    int_to_vals, vals_to_int = get_mappers(df)

    d = df.set_index(KEYS).to_dict()[METRIC]

    return d, df, int_to_vals, vals_to_int

class SimulatedAnnealing:
    def __init__(self, data, int_to_vals, vals_to_int):
        self.data = data #dataframe with full parameter space -> relax condition
        self.int_to_vals = int_to_vals
        self.vals_to_int = vals_to_int

    def set_seed(self, seed=0):
        np.random.seed(seed)

    def init_temperature(self, N_iter=10, pick_max=False):
        if pick_max:
            return np.max(list(self.data.values())) #worst-case

        keys = np.random.randint(0, len(self.data), size=N_iter)
        
        key_list = list(self.data.keys())

        return np.median([self.data[key_list[k]] for k in keys])

    def init_params(self, pick_max=False):
        if pick_max:
            max_val = self.init_temperature(pick_max=True)
            max_key = [x for x in self.data if self.data[x]==max_val]
            
            return max_key[0]

        return list(self.data.keys())[np.random.randint(0, len(self.data))] # (ITR, DVFS, RAPL) initial value

    def neighbor(self, params):
        neighbor_list = find_neighbor_keys(params, self.int_to_vals, self.vals_to_int)

        return neighbor_list[np.random.randint(0, len(neighbor_list))]

    def iterate(self,
                N_iter, 
                init_temp, 
                decay_factor=0.99, 
                decay_N_iter=100,
                N_display=100,
                pick_max=False):

        self.records = []
        self.temperature = []

        temperature = init_temp
        self.temperature.append(temperature)

        params_current = self.init_params(pick_max=pick_max) #(ITR, DVFS, RAPL) = (16, 3072, 95)
        obj_current = self.data[params_current] #metric of interest (EDP, Power etc.)
        print(f'Initial Objective = {obj_current}')

        self.params_min = params_current
        self.obj_min = obj_current

        for i in range(N_iter): #start iterating
            params_candidate = self.neighbor(params_current) #(16, 3080, 75) - same message size

            if i % decay_N_iter == 0:
                temperature *= decay_factor
            self.temperature.append(temperature)

            obj_candidate = self.data[params_candidate] #(EDP, Time, Energy) for neighbor
            #obj_candidate = do_experiment(params_candidate) #-> call test-bed with the parameters, do the experiment and return the value

            #---------(Current Params, New Params -> Current EDP, New EDP)
            prob = np.min([np.exp((obj_current - obj_candidate) / temperature), 1])

            if np.random.uniform() < prob:
                params_current = params_candidate

                obj_current = obj_candidate

            #keep track of the minimum value
            if obj_current < self.obj_min:
                self.obj_min = obj_current
                self.params_min = params_current

            self.records.append(obj_current)

            if i % N_display == 0:
                print(f'Current Value = {obj_current} with Params = {params_current}')

        return params_current

    def plot(self, msg, save_name):
        if not hasattr(self, "records"):
            raise ValueError("Please run iterate before plotting")

        N_data = len(self.data)

        plt.figure()
        plt.plot(np.sort(list(self.data.values())), label='EDP (Sorted)')
        plt.plot(self.records[0:N_data], label='Optim Evolution')
        plt.xlabel('Iteration Number (Truncated to Number of Data Points)')
        plt.ylabel('EDP')
        plt.title(f'Evolution of Simulated Annealing on Msg Size = {msg}')

        plt.savefig(save_name)

def run(N_iter=10000, msg_size=65536, worst_start=False, output_plot_fname='plot.png'):
    d, df, int_to_vals, vals_to_int = prepare_data('6_14.csv', msg_size=msg_size)
    sa = SimulatedAnnealing(d, int_to_vals, vals_to_int) 

    if worst_start:
        pick_max = True
    else:
        pick_max = False

    sa.iterate(N_iter, init_temp=sa.init_temperature(pick_max=pick_max), pick_max=pick_max)
    sa.plot(msg_size, output_plot_fname)

    return sa

class RandomSearch:
    def __init__(self, data):
        self.data = data

class CEM:
    def __init__(self, data):
        self.data = data

'''
    612 1000
    612 12288
    612 131072
    612 1321
    612 16384
    612 24576
    612 4096
    612 49152
    612 7777
    711 8192
    612 98304
    612 9999

'''

def compute_features_from_all_logs(loc, N_parallel=1):
    files = glob.glob(f'{loc}/*.csv')
    msgs = np.unique([f.split('/')[-1].split('.')[3].split('_')[2] for f in files])

    df_list = []
    for msg in msgs:
        print(f"Constructing features for msg = {msg}...")
        df = compute_features_from_logs(f'{loc}/*{msg}*.csv', N_parallel=N_parallel)
        df.reset_index(inplace=True)
        df_list.append(df)

    df = pd.concat(df_list, axis=0)

    return df



def compute_features_from_logs(loc, N_parallel=4):
    def compute_entropy(df, colname, nonzero=False):
        x  = df[colname].value_counts()

        if nonzero:
            x = x[x.index > 0]

        x = x / x.sum() #scipy.stats.entropy actually automatically normalizes

        return entropy(x)

    def featurize(file, data):
        df = pd.read_csv(file, sep=' ')
        df_non0j = df[df['joules'] > 0].copy()

        #non-zero joules
        df_non0j['timestamp'] = df_non0j['timestamp'] - df_non0j['timestamp'].min()
        df_non0j['joules'] = df_non0j['joules'] - df_non0j['joules'].min()
        
        df_non0j['timestamp'] = df_non0j['timestamp'] * TIME_CONVERSION
        df_non0j['joules'] = df_non0j['joules'] * JOULE_CONVERSION

        #add features here
        d = {'name': file}        
        for col in ['rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes']:
            d[f'entropy_{col}'] = compute_entropy(df, col, nonzero=False)
            d[f'entropy_nonzero_{col}'] = compute_entropy(df, col, nonzero=True)

        tags = file.split('.')[-2]
        rnd, cpu, msg, nrounds, itr, dvfs, rapl = tags.split('_')
        d['rnd'] = int(rnd)
        d['cpu'] = int(cpu)
        d['msg'] = int(msg)
        d['nrounds'] = int(nrounds)
        d['itr'] = int(itr)
        d['dvfs'] = dvfs
        d['rapl'] = int(rapl)

        last_row = df_non0j.tail(1).iloc[0]
        d['joules'] = last_row['joules']
        d['time'] = last_row['timestamp']
        d['edp'] = 0.5 * d['joules'] * d['time']
        d['n_interrupts'] = df.shape[0]
        d['n_nonzero_interrupts'] = df[df['rx_bytes']>0].shape[0]

        cols = ['rx_desc', 'tx_desc', 'rx_bytes', 'tx_bytes']
        corr = df[cols].corr()
        for i in range(len(cols)):
            for j in range(i):
                d[f'corr_{cols[i]}_{cols[j]}'] = corr[cols[i]][cols[j]]

        data.append(d)

    manager = Manager()
    data = manager.list()
    plist = []

    N_current = 0
    for file in glob.glob(f'{loc}'):

        p = Process(target=featurize, args=(file, data))
        p.start()
        N_current += 1

        plist.append(p)

        if N_current % N_parallel == 0:
            [p.join() for p in plist]

    data = list(data)
    data = pd.DataFrame(data)
    data.set_index(['name', 'rnd', 'cpu', 'msg', 'nrounds', 'itr', 'dvfs', 'rapl'], inplace=True)
    
    return data

def fit_pred(df, max_depth):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    msgs = df['msg'].unique()
    entropy_cols = [c for c in df.columns if c.find('entropy')>-1]
    corr_cols = [c for c in df.columns if c.find('corr_')>-1]

    for msg in msgs:
        print(f'\nMsg = {msg}')
        df_train, df_test = train_test_split(df, train_size=0.7)

        model = RandomForestRegressor(max_depth=max_depth)
        model.fit(df_train[entropy_cols + corr_cols], df_train['edp'])

        train_pred = model.predict(df_train[entropy_cols + corr_cols])
        train_target = df_train['edp']

        test_pred = model.predict(df_test[entropy_cols + corr_cols])
        test_target = df_test['edp']

        print(f'Train : {((train_pred - train_target)**2).mean()}')
        print(f'Test  : {((test_pred - test_target)**2).mean()}')

'''
Tx/Rx bytes:
    Entropy
    Correlations
    Interrupt time-differences?
    Reward = \Sigma edp per time-step -> r1 + r2 + ... + rT
'''
class PolicyGradient:
    def __init__(self, data, N_inputs, N_nodes, N_layers, activation, output_activation, df):
        self.data = data

        self.policy = PolicyNet(N_inputs, N_nodes, N_layers, activation, output_activation, df)

        '''
        Options:
        Summarizes data (features) -> Action (ITR)

        Raw log files (RNN) -> Action (ITR)

        Features/Log files look up and Architecture
        '''

    def training_loop(self,
                      N_iter, 
                      batch_size, 
                      env,
                      policy=None, 
                      lr=1e-2,
                      critic_lr=1e-1, 
                      causal=False,
                      baseline=False,
                      critic=None,
                      critic_update='MC',
                      debug=False):
        
        if policy is None:
            policy = PolicyNet(N_inputs, N_outputs, N_hidden_layers, N_hidden_nodes, nn.ReLU(), nn.Sigmoid())
        
        reward_curve = {}

        optimizer_policy = optim.Adam(policy.parameters(), lr=lr)
        if critic:
            optimizer_critic = optim.Adam(critic.parameters(), lr=critic_lr)

        exp_reward_list = []

        for i in range(N_iter):
            #step 1: generate batch_size trajectories
            J_policy, mean_reward, J_critic = create_trajectories(env, policy, batch_size, causal=causal, baseline=baseline, critic=critic, critic_update=critic_update)
            #print(f'Critic Loss = {J_critic}')
            
            #step 2: define J and update policy
            optimizer_policy.zero_grad()
            (-J_policy).backward()
            optimizer_policy.step()

            #step 3: 
            if critic:
                if J_critic.item() > 0.1:
                    optimizer_critic.zero_grad()
                    J_critic.backward()
                    optimizer_critic.step()        

            if i % 10 == 0:
                print(f"Iteration {i} : Mean Reward = {mean_reward}")
                reward_curve[i] = mean_reward

        return reward_curve
   
    def create_trajectories(self,
                            env, 
                            policy, 
                            N, 
                            causal=False, 
                            baseline=False, 
                            critic=False, 
                            critic_update='MC'):

        action_probs_all_list = []
        rewards_all_list = []
        states_all_list = []

        for _ in range(N): #run env N times
            state = env.reset()
            
            action_prob_list, reward_list, state_list = torch.tensor([]), torch.tensor([]), torch.tensor([])
            done = False 

            while not done:
                state_list = torch.cat((state_list, torch.tensor([state]).float())) #

                action_probs = policy(torch.from_numpy(state).unsqueeze(0).float()).squeeze(0)

                action_selected_index = torch.multinomial(action_probs, 1)
                action_selected_prob = action_probs[action_selected_index]
                action_selected = action_space[action_selected_index]

                state, reward, done, info = env.step(action_selected.item())

                action_prob_list = torch.cat((action_prob_list, action_selected_prob))
                reward_list = torch.cat((reward_list, torch.tensor([reward])))

            action_probs_all_list.append(action_prob_list)
            rewards_all_list.append(reward_list)
            states_all_list.append(state_list)

        #non-optimized code (negative strides not supported by torch yet)
        rewards_to_go_list = [torch.tensor(np.cumsum(traj_rewards.numpy()[::-1])[::-1].copy())
                              for traj_rewards in rewards_all_list]


        #compute objective
        J = 0
        critic_inputs, critic_targets = [], []
        #for clarity, refactoring code below into specific modes. some of these can be combined
        if not baseline and not causal:
            print("No Baseline - Not Causal")
            for idx in range(N):
                J += action_probs_all_list[idx].log().sum() * (rewards_all_list[idx].sum())

        if not baseline and causal:
            print("No Baseline - Causal")
            for idx in range(N):
                row = rewards_to_go_list[idx]
                J += (action_probs_all_list[idx].log() * row).sum()

        #critic only shows up in baseline cases
        if baseline and not causal and not critic: #critic only affects baseline cases
            print("Baseline - Not Causal, No Critic")
            baseline_term = np.mean([torch.sum(r).item() for r in rewards_all_list])

            for idx in range(N):
                J += action_probs_all_list[idx].log().sum() * (rewards_all_list[idx].sum() - baseline_term)

        if baseline and not causal and critic:
            raise NotImplementedError("Probably not useful")
            for idx in range(N):
                row = rewards_to_go_list[idx]
                actions = action_probs_all_list[idx]
                states = states_all_list[idx]

                T = len(row)
                for t in range(T):
                    J += actions[t].log() * (row[t] - critic(torch.cat([torch.tensor([t]).float(), states[t]])))

        if baseline and causal and not critic:
            '''Need time-dependent baseline terms
            '''
            print("Baseline - Causal, No Critic")
            #compute time-dependent baseline terms (mean reward to go)
            T = np.max([len(row) for row in rewards_to_go_list])
            baseline_term = torch.zeros(N, T)
            for idx in range(N):
                row = rewards_to_go_list[idx]
                baseline_term[idx] = torch.cat((row, torch.zeros(T-len(row)))) #pad with 0s if episode took time < T (end)
            baseline_term = torch.mean(baseline_term, dim=0) #time-dependent means

            #compute J
            for idx in range(N):
                row = rewards_to_go_list[idx]
                J += (action_probs_all_list[idx].log() * (row - baseline_term[:len(row)])).sum() #subtract time-dependent means

        if baseline and causal and critic:
            #train/update critic
            #(state, t) -> (rewards to go)

            #print("Baseline - Causal, Critic")
            #return states_all_list, action_probs_all_list, rewards_to_go_list
            for idx in range(N):
                row = rewards_to_go_list[idx]
                rewards = rewards_all_list[idx] #needed to update critic
                actions = action_probs_all_list[idx]
                states = states_all_list[idx]

                T = len(row)
                for t in range(T):
                    critic_in = torch.cat([torch.tensor([t]).float(), states[t]])
                    
                    J += actions[t].log() * (row[t] - critic(critic_in))

                    if critic_update=='MC':
                        critic_inputs.append(critic_in)
                        critic_targets.append(row[t]) #pure MC - target = reward to go
                    
                    if critic_update=='TD':
                        if t < T-1:
                            critic_inputs.append(critic_in)
                            critic_targets.append(rewards[t].unsqueeze(0) + critic(torch.cat([torch.tensor([t+1]).float(), states[t+1]]))) #TD - target = current reward + critic estimate
        J = J / N


        R = np.mean([torch.sum(r).item() for r in rewards_all_list]) #track overall progress

        #critic loss computation
        J_critic = None
        critic_loss = nn.MSELoss()
        
        if critic:
            critic_inputs = torch.stack(critic_inputs)
            critic_targets = torch.stack(critic_targets)

            preds = critic(critic_inputs)

            J_critic = critic_loss(preds, critic_targets)

        return J, R, J_critic    


    def train(self, N_batch_size, T=10):
        for exp in range(N_batch_size):

            for t in range(T):
                pass
                #step 1: construct inputs
                #step 2: predict action probs
                #step 3: sample from multinomial distribution (everything discrete here)
                #step 4: measure instantaneous reward

            #step 5:

    def get_learning_curves(self,
                            N_exp, 
                            N_iter, 
                            batch_size, 
                            env, 
                            causal=False, 
                            baseline=False):

        reward_curve_list = []
        for _ in range(N_exp):
            policy = PolicyNet(N_inputs, N_outputs, N_hidden_layers, N_hidden_nodes, activation, output_activation)

            reward_curve = training_loop(N_iter, batch_size, env, policy, causal=causal, baseline=baseline)

            reward_curve_list.append(reward_curve)

        #combine results
        rcurve = {}
        for k in reward_curve_list[0]:
            rcurve[k] = [r[k] for r in reward_curve_list]
            rcurve[k] = (np.mean(rcurve[k]), np.std(rcurve[k]))

        return rcurve

    def plot_learning_curve(self,
                            rcurve, 
                            label):

        k = list(rcurve.keys())

        plt.errorbar(k, [rcurve[key][0] for key in k], [rcurve[key][1] for key in k], label=label)
        plt.legend()
        plt.xlabel('Episode Number')
        plt.ylabel('Average Score')
        plt.title('10 runs, 200 iterations each, batch-size 20 episodes')
        #plt.savefig('LearningCurve.png')



#map msg size -> (itr, rapl, dvfs)
'''
This is the neural network that acts as the policy.

The policy should take the state of the system and suggest actions.

State/Inputs: There are many possibilities
    1. Current incoming message size or last K message sizes
    2. Statistics from last K sends and receive. 
       Entropy of sent bytes or correlation coefficients for e.g.
    3. Entries from log files. This would require a different architecture
       instead of the feed-forward architecture below. Since log entries are to
       be processed sequentially, we would use a recurrent neural network (LSTMs for e.g.)

Actions/Outputs:
    The core of the neural network is shared in the sense that inputs are propagated
    through multiple layers. At the very end, three streams are peeled off - one for
    each of the output values. In our case, this can be ITR, RAPL and DVFS.

    For now, let's focus just on ITR.

    An important note: we don't predict the exact values of the ITR. Instead the policy
    predicts a probability distribution over all possible ITR values. We can then either
    pick the one with the maximum probability or sample from the given distribution (i.e.
    pick a value according to the probabilities which might give ITR values that don't have
    the highest probability). Sampling is generally better. It allows us to explore paths
    in the parameter space.

'''
class PolicyNet(nn.Module):
    def __init__(self, 
                 N_inputs, 
                 N_outputs, 
                 N_fields,
                 N_nodes, 
                 N_layers, 
                 activation, 
                 output_activation):

        super(PolicyNet, self).__init__()

        #msg_embedding_dim = 4
        #N_itr = len(df['ITR'].unique())
        #N_rapl = len(df['RAPL'].unique())
        #N_dvfs = len(df['DVFS'].unique())

        if not isinstance(N_outputs, tuple):
            raise ValueError('Expected N_outputs to be a tuple')
            if len(N_outputs) != N_fields: #number of registers we are predicting
                raise ValueError(f'Expected N_outputs to have {N_fields} fields')

        self.activation = activation
        self.output_activation = output_activation

        #self.emb_msg = nn.Embedding(4, msg_embedding_dim)

        self.layers = nn.ModuleList()
        for i in range(N_layers):
            if i==0:
                #self.layers.append(nn.Linear(msg_embedding_dim, N_nodes))
                self.layers.append(nn.Linear(N_inputs, N_nodes))
            else:
                self.layers.append(nn.Linear(N_nodes, N_nodes))

        self.output_layers = nn.ModuleList()
        for i in range(N_fields):
            self.output_layers.append(nn.Linear(N_nodes, N_outputs[i]))
    
        #self.output_rapl = nn.Linear(N_nodes, N_rapl)
        #self.output_dvfs = nn.Linear(N_nodes, N_dvfs)

    def forward(self, inp):
        #embedding
        #o = self.emb_msg(inp)
        o = inp

        #hidden layers
        for layer in self.layers:
            o = self.activation(layer(o))

        #output
        output = torch.zeros(len(self.output_layers))
        for l in self.output_layers:
            output[l] = self.output_activation(self.output_layers[l](o))

        #pred_itr = self.output_activation(self.output_itr(o))
        #pred_rapl = self.output_activation(self.output_rapl(o))
        #pred_dvfs = self.output_activation(self.output_dvfs(o))

        #return pred_itr, pred_rapl, pred_dvfs
        return output