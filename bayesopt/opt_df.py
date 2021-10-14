import read_agg_data
from dragonfly import load_config, minimize_function

import numpy as np
import pandas as pd
from collections import namedtuple
import matplotlib.pylab as plt
from matplotlib import gridspec
import itertools
import pickle
import pdb
import os

plt.ion()
global_accesses = {}

key = namedtuple('key', ['workload', 'sys', 'conf', 'n_init_points', 'kappa', 'n_exp'])

def run(debug=False):
    #config = {'netpipe': [('msg', 64), ('msg', 8192), ('msg', 65536), ('msg', 524288)],
    #          'nodejs': [None]
    #        }
    config = {'netpipe': [('msg', 64)]}

    results = {}

    for workload in ['netpipe']:
        df_comb, _, _ = read_agg_data.start_analysis(workload) #DATA
        df_comb['dvfs'] = df_comb['dvfs'].apply(lambda x: int(x, base=16))
        df_comb = df_comb[(df_comb['itr']!=1) | (df_comb['dvfs']!=65535)] #filter out linux dynamic

        for sys in ['linux_tuned', 'ebbrt_tuned']:
            df = df_comb[(df_comb['sys']==sys)].copy()

            #depends on sys
            INDEX_COLS = [] #what variables to search over
            uniq_val_dict = {} #range for each variable
            domain = [] #dragonfly domain

            for colname in ['itr', 'dvfs', 'rapl']:
                uniq_vals = np.sort(df[colname].unique())
                
                if len(uniq_vals) > 1:
                    INDEX_COLS.append(colname)
                    uniq_val_dict[colname] = uniq_vals #all unique values

                    domain.append({'name': colname, 'type': 'discrete_numeric', 'items': uniq_vals, 'dim': 1})


            if debug:
                print(f'Workload:\n{workload}\n------')
                print(f'INDEX_COLS:\n{INDEX_COLS}\n------')
                print(f'domain:\n{domain}\n------')
                print(f'uniq_val_dict:\n{uniq_val_dict}\n------')

            df.set_index(INDEX_COLS, inplace=True) #df for fixed workload, for fixed OS

            for conf in config[workload]:
                if conf is not None:
                    df_lookup = df[df[conf[0]]==conf[1]]
                else:
                    df_lookup = df.copy()

                if debug:
                    print(f'df_lookup:\n{df_lookup.head()}\n------')

                obj = lambda arr: objective(arr, df_lookup, INDEX_COLS)
                for n_init_points in [2, 5, 10]: #randomly sampled points
                    for n_iter in [30]: #total iterations                
                        #for utility in ['acq']:
                        for kappa in [1, 5, 10, 20]: #kappa values
                            for n_exp in range(1): #number of trials                            
                                print(f'------{workload} {conf} n_init={n_init_points} n_iter={n_iter} kappa={kappa} n_exp={n_exp}-------')
                                k = key(workload, sys, conf, n_init_points, kappa, n_exp)

                                conf = load_config({'domain': domain})
                                val, point, history = minimize_function(obj, conf.domain, n_iter, config=conf)
                                                                                
                                results[k] = (history.query_points, history.query_vals, df_lookup['edp_mean'].min(), df_lookup.shape[0], df_lookup['edp_mean'].sort_values())

    return results

def objective(arr, df, INDEX_COLS):
    arr = [val[0] for val in arr] #since dragonfly wraps each val with dim=1 by default

    print(arr)

    if len(arr)==2:
        val = df.loc[arr[0], arr[1]].edp_mean
    else:
        val = df.loc[arr[0], arr[1], arr[2]].edp_mean

    return val

def save_results(r, filename):
    s = {}
    for key in r:
        s[key] = (r[key][0].res, *r[key][1:])

    pickle.dump(s, open(filename, 'wb'))

def read_results(filename):
    return pickle.load(open(filename, 'rb'))

def plot(r, tag, loc=None):
    for k in r:
        plt.figure(figsize=(10,7))
        plt.plot(np.array(r[k][4]))
        plt.xlabel('Unique parameter configuration')
        plt.ylabel('EDP')
        
        bayesopt_min = np.min(r[k][1])
        global_min = r[k][2]
        plt.title(f'{str(k)}\nglobal_min={global_min:.2f} bayesopt_min={bayesopt_min:.2f}')

        N = r[k][3]
        for elem in r[k][1]:        
            plt.plot(np.arange(N), [elem]*N, color='r')

        if loc is not None:
            if not os.path.exists(loc):
                os.makedirs(loc)

            #hard-coded for now
            dirname = f"{getattr(k, 'workload')}_{str(getattr(k, 'conf'))}_{getattr(k, 'sys')}"
            if not os.path.exists(f'{loc}/{dirname}'):
                os.makedirs(f'{loc}/{dirname}')

            filename = f'{tag}_'
            for f in k._fields:
                filename += f'{f}{getattr(k, f)}_'
            filename = filename.rstrip('_') + '.png'

            plt.savefig(os.path.join(loc, dirname, filename))
            plt.close()