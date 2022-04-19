import read_agg_data
from dragonfly import load_config, minimize_function, multiobjective_minimize_functions

import numpy as np
import pandas as pd
from collections import namedtuple

import matplotlib.pylab as plt
from matplotlib import gridspec
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import itertools
import pickle
import pdb
import os

plt.ion()
global_accesses = {}

'''
1. Read (itr, dvfs, rapl, qps) lines from mcd data

2. A function to return result of an experiment

3. Training loop

4. Plotting

a. metric is sorted, red lines for experiments/lookups
b. x-axis: time/experiment number, y-axis: current best value

5. one optimal set for each qps
a. run bayesopt for 10 representative qps value
   interpolate between them
   if qps=250k, maybe(?) itr/dvfs for closer qps

'''

key = namedtuple('key', ['workload', 'sys', 'conf', 'n_init_points', 'n_exp'])

def run(debug=False, fix_rapl=135, metric='read_99th_mean'):
    config = {'netpipe': [('msg', 64), ('msg', 8192), ('msg', 65536), ('msg', 524288)],
              'nodejs': [None],
              'mcd': [('target_QPS', 200000)] #, ('target_QPS', 400000), ('target_QPS', 600000)],
              #'mcdsilo': [('target_QPS', 200000), ('target_QPS', 500000), ('target_QPS', 600000)],
            }
    #config = {'netpipe': [('msg', 64)]}

    results = {}
    failures = {}

    for workload in ['mcd']:
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
                
                if fix_rapl and colname=='rapl':
                    df = df[df['rapl']==fix_rapl].copy()
                    continue

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

                obj = lambda arr: objective(arr, df_lookup, INDEX_COLS, metric=metric)
                #obj1 = lambda arr: objective(arr, df_lookup, INDEX_COLS, metric='joules_mean')

                for n_init_points in [5]: #randomly sampled points
                    for n_iter in [30]: #total iterations
                        for n_exp in range(1): #number of trials
                            
                            print(f'------{workload} {conf} n_init={n_init_points} n_iter={n_iter} n_exp={n_exp}-------')
                            k = key(workload, sys, conf, n_init_points, n_exp)

                            config_df = load_config({'domain': domain})

                            try:
                                val, point, history = minimize_function(obj, config_df.domain, n_iter, config=config_df)
                                #val, point, history = multiobjective_minimize_function([obj, obj1], config_df.domain, n_iter, config=config_df)
                            except:                                
                                failures[k] = 1
                                continue

                            results[k] = (history.query_points, history.query_vals, df_lookup[metric].min(), df_lookup.shape[0], df_lookup[metric].sort_values(), df_lookup[metric])
                            

    return results, failures

def objective(arr, df, INDEX_COLS, metric='read_99th_mean'):
    arr = [val[0] for val in arr] #since dragonfly wraps each val with dim=1 by default

    print(arr)

    if len(arr)==2:
        val = df.loc[arr[0], arr[1]][metric]
    else:
        val = df.loc[arr[0], arr[1], arr[2]][metric]

    return val

def save_results(r, filename):
    s = {}
    for key in r:
        s[key] = (r[key][0].res, *r[key][1:])

    pickle.dump(s, open(filename, 'wb'))

def read_results(filename):
    return pickle.load(open(filename, 'rb'))

'''read_99th_mean
def plot_surface(r_val):
    pts = r[0]
    vals = r[1]

    x = [i[0][0] for i in pts]
    y = [i[0][1] for i in pts]    

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X**2 + Y**2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
'''

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