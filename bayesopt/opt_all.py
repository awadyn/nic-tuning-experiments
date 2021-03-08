import read_agg_data
from bayes_opt import BayesianOptimization, UtilityFunction

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib import gridspec
import itertools
import pdb

plt.ion()

def run(debug=False):
    config = {'netpipe': [('msg', 64), ('msg', 8192), ('msg', 65536), ('msg', 524288)],
              'nodejs': [None]
            }
    results = {}

    for workload in ['netpipe', 'nodejs']:
        df_comb, _, _ = read_agg_data.start_analysis(workload) #DATA
        df_comb['dvfs'] = df_comb['dvfs'].apply(lambda x: int(x, base=16))
        df_comb['edp_mean'] *= -1
        df_comb = df_comb[(df_comb['itr']!=1) | (df_comb['dvfs']!=65535)] #filter out linux dynamic

        for sys in ['linux', 'ebbrt']:
            df = df_comb[(df_comb['sys']==sys)].copy()

            #depends on sys
            INDEX_COLS = []
            domain_dict = {}
            uniq_val_dict = {}
            for colname in ['itr', 'dvfs', 'rapl']:
                uniq_vals = np.sort(df[colname].unique())
                
                if len(uniq_vals) > 1:
                    INDEX_COLS.append(colname)
                    domain_dict[colname] = (uniq_vals[0], uniq_vals[-1]) #(min, max)
                    uniq_val_dict[colname] = uniq_vals #all unique values

            if debug:
                print(f'Workload:\n{workload}\n------')
                print(f'INDEX_COLS:\n{INDEX_COLS}\n------')
                print(f'domain_dict:\n{domain_dict}\n------')
                print(f'uniq_val_dict:\n{uniq_val_dict}\n------')

            df.set_index(INDEX_COLS, inplace=True)

            for conf in config[workload]:
                if conf is not None:
                    df_lookup = df[df[conf[0]]==conf[1]]
                else:
                    df_lookup = df.copy()

                if debug:
                    print(f'df_lookup:\n{df_lookup.head()}\n------')

                obj = lambda **kwargs: objective_interp(df_lookup, INDEX_COLS, uniq_val_dict, debug=debug, **kwargs)
                for n_init_points in [2, 5, 10]: #randomly sampled points
                    for n_iter in [30]: #total iterations                
                        #for utility in ['acq']:
                        for kappa in [1, 5, 10, 20]: #kappa values
                            for n_exp in range(1): #number of trials                            
                                print(f'------{workload} {conf} n_init={n_init_points} n_iter={n_iter} kappa={kappa} n_exp={n_exp}-------')
                                optimizer = BayesianOptimization(obj, domain_dict) #randomly generate points
                                optimizer.maximize(init_points=n_init_points, n_iter=n_iter, kappa=kappa)
            
                                results[(workload, sys, conf, n_init_points, kappa, n_exp)] = (optimizer, df_lookup['edp_mean'].max(), df_lookup.shape[0], df_lookup['edp_mean'].sort_values())

    return results

def objective_interp(df, index_cols, uniq_val_dict, debug=False, **kwargs):
    def compute_alpha(limit_list, val):
        low = limit_list[0]
        high = limit_list[1]
        assert(high > low)

        return (val - low) / (high - low)

    endpoints = {}
    alpha = {}
    low_high_ind = {}
    for idx in index_cols:
        uniq_vals = uniq_val_dict[idx]

        endpoints[idx] = np.sort(uniq_vals[np.argsort(np.abs(uniq_vals - kwargs[idx]))][0:2]) #FIX THIS

        alpha[idx] = compute_alpha(endpoints[idx], kwargs[idx])

        low_high_ind[idx] = {}
        low_high_ind[idx][endpoints[idx][0]] = 0
        low_high_ind[idx][endpoints[idx][1]] = 1

    space = list(itertools.product(*[endpoints[idx] for idx in index_cols]))

    coefficients = []
    for elem in space:
        coef_list = []
        for idx in range(len(index_cols)):
            index = index_cols[idx]
            ind = low_high_ind[index][elem[idx]]

            if ind == 0: #can replace with linear combination
                coef_list.append(1-alpha[index])
            else: 
                coef_list.append(alpha[index])
        coefficients.append(np.prod(coef_list))
    assert(np.abs(np.sum(coefficients)-1)<10-5)
    assert(len(space)==len(coefficients)) #by construction - can remove

    if debug:
        print(f'kwargs:\n{kwargs}\n------')
        print(f'endpoints:\n{endpoints}\n------')
        print(f'alpha:\n{alpha}\n------')
        print(f'space:\n{space}\n------')
        print(f'low_high_ind:\n{low_high_ind}\n------')

        '''
        alpha_itr = alpha['itr']
        alpha_dvfs = alpha['dvfs']
        alpha_rapl = alpha['rapl']
        coefficients2 = [(1-alpha_itr)*(1-alpha_dvfs)*(1-alpha_rapl),
                        (1-alpha_itr)*(1-alpha_dvfs)*alpha_rapl,
                        (1-alpha_itr)*alpha_dvfs*(1-alpha_rapl),
                        (1-alpha_itr)*alpha_dvfs*alpha_rapl,
                        alpha_itr*(1-alpha_dvfs)*(1-alpha_rapl),
                        alpha_itr*(1-alpha_dvfs)*alpha_rapl,
                        alpha_itr*alpha_dvfs*(1-alpha_rapl),
                        alpha_itr*alpha_dvfs*alpha_rapl                    
                       ]

        print(coefficients)
        print(coefficients2)
        '''

    edp_avg = 0
    for idx in range(len(space)):
        corner = space[idx]

        if debug: print(corner, df.head())
        try:
            assert(isinstance(df.loc[corner], pd.Series))
        except:
            pdb.set_trace()

        edp_val = df.loc[corner].loc['edp_mean'] #METRIC
        if debug: print(corner, coefficients[idx], edp_val)

        edp_avg += coefficients[idx] * edp_val

    return edp_avg