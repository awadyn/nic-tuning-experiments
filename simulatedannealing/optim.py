import pandas as pd
import numpy as np
import matplotlib.pylab as plt

plt.ion()

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

class PolicyGradient:
    def __init__(self, data):
        self.data = data

