import numpy as np
import itertools
from pprint import pprint
import matplotlib.pylab as plt
import pandas as pd
import os

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render, init_notebook_plotting

from ax.service.managed_loop import optimize

from ax.metrics.branin import branin

from ax.utils.measurement.synthetic_functions import hartmann6

import read_agg_data

config = {
            'mcd': [('target_QPS', 200000) , ('target_QPS', 400000), ('target_QPS', 600000)]
         }

def prepare_data(fix_rapl=135):
    df_comb, _, _ = read_agg_data.start_analysis('mcd') #DATA                                                  
    df_comb['dvfs'] = df_comb['dvfs'].apply(lambda x: int(x, base=16))
    df_comb = df_comb[(df_comb['itr']!=1) | (df_comb['dvfs']!=65535)] #filter out linux dynamic    
    
    df_dict = {}
    for sys in ['linux_tuned', 'ebbrt_tuned']:
        df = df_comb[(df_comb['sys']==sys)].copy()
        
        for conf in config['mcd']:
            if conf is not None:
                df_lookup = df[df[conf[0]]==conf[1]]
            else:
                df_lookup = df.copy()
                    
            
            INDEX_COLS = [] #what variables to search over                                                            
            uniq_val_dict = {} #range for each variable                                                               

            for colname in ['itr', 'dvfs', 'rapl']:
                uniq_vals = np.sort(df_lookup[colname].unique())

                if fix_rapl and colname=='rapl':
                    df_lookup = df_lookup[df_lookup['rapl']==fix_rapl].copy()
                    continue
    
                if len(uniq_vals) > 1:
                    INDEX_COLS.append(colname)
                    uniq_val_dict[colname] = uniq_vals #all unique values                                 
                
            df_lookup.set_index(INDEX_COLS, inplace=True) #df for fixed workload, for fixed OS                               
            
            df_dict[(sys, conf[1])] = (df_lookup, INDEX_COLS, uniq_val_dict)
            
    return df_dict

def prepare_search_space(df, var_type='discrete'):
    if var_type != 'discrete' and var_type != 'range':
        raise ValueError('var_type should be "discrete" or "range"')

    idx = df.index
    search_space = []
    
    for i, name in enumerate(idx.names):
        if var_type=='range':
            s = {'name': name,
                 'type': 'range',
                 'bounds': [np.min([val[i] for val in idx]), np.max([val[i] for val in idx])],
                 'value_type': 'int',
                 'log_scale': False
                }
            search_space.append(s)
        
        elif var_type=='discrete':
            s = {'name': name,
                 'type': 'choice',
                 'values': list(np.unique([val[i] for val in idx])),
                 'value_type': 'float',
                 'log_scale': False,
                 'is_ordered': True,
                 #'sort_values': True
                }
            search_space.append(s)
                        
    return search_space

def mcd_eval_func(params, 
                  df, index_cols, 
                  uniq_val_dict, 
                  metric,
                  fill_missing=False, 
                  missing_val=None):
    
    x = [params.get(idx) for idx in index_cols]
    
    #no noise on either val or l2norm
    try:
        res = {
                'mcd': (df[metric].loc[tuple(x)], 0.0)
              }
    except:
        if not fill_missing:
            raise ValueError(f"Cannot find key {x}")
        else:
            if missing_val is None:
                raise ValueError(f"Please pass missing_val")
                                 
            res = {
                    'mcd': (missing_val, 0.0)
                  }
        
    return res
        
def missing_keys(df):
    d = df[2]
    df = df[0]
    
    missing = []
    
    for i in d['itr']:
        for j in d['dvfs']:
            try:
                df.loc[i,j]['joules_mean']
            except:
                missing.append([i,j])
                               
    return missing

def list_missing_keys(df_dict):
    for k in df_dict:

        missing = missing_keys(df_dict[k])
        print(k)
        print(missing, '\n')
        
def get_nbs(tune_vals):
    tune_nbs = {}
    n_itr = len(tune_vals)

    for idx, val in enumerate(tune_vals):
        low, high = None, None

        if idx>0: low = tune_vals[idx-1]
        if idx<n_itr-1: high = tune_vals[idx+1]

        tune_nbs[val] = (low, high)

    return tune_nbs  


def collect_nb_dataset(df_lookup, uniq_val_dict, metric='joules_mean', debug=False):
    target_vals = []
    nb_vals = []

    itr_nbs = get_nbs(uniq_val_dict['itr'])
    dvfs_nbs = get_nbs(uniq_val_dict['dvfs'])
    
    for itr in uniq_val_dict['itr']:
        for dvfs in uniq_val_dict['dvfs']:
            try:
                df_lookup.loc[itr, dvfs]

                #find neighbors of (itr, dvfs) -> (itr+/-1, dvfs), (itr, dvfs+/-1)
                val_dict = {}
                if debug:print('----')
                if debug: print(itr, dvfs)
                for itr_n in itr_nbs[itr]:
                    if itr_n is not None:
                        if debug: print(itr_n, dvfs)
                        val = df_lookup.loc[itr_n, dvfs][metric]
                        val_dict[(itr_n, dvfs)] = val

                for dvfs_n in dvfs_nbs[dvfs]:
                    if dvfs_n is not None:
                        if debug: print(itr, dvfs_n)
                        val = df_lookup.loc[itr, dvfs_n][metric]
                        val_dict[(itr, dvfs_n)] = val

                if debug: print(val_dict)

                target_vals.append((itr, dvfs, df_lookup.loc[itr, dvfs][metric]))
                nb_vals.append(val_dict)
            except: #if missing, don't use
                continue

    return target_vals, nb_vals    

def check_simple_interp(target_vals, nb_vals):
    assert(len(target_vals)==len(nb_vals))
    
    actuals = [t[2] for t in target_vals]
    preds = [np.mean(list(n.values())) for n in nb_vals]
    
    plt.figure()
    plt.plot(preds, actuals, 'p')
    plt.xlabel('preds')
    plt.ylabel('actuals')
    plt.plot(preds, preds) #diagonal
    
    error = np.sqrt(np.mean((np.array(actuals)/np.array(preds) - 1)**2))
    plt.title(f'Simple Interp (mean of nbs): RMS of percentage error = {100*error:.3f}%')
    
_ = '''
Note on interpolation:

We have a grid of (itr, dvfs) points. The spacings along itr are not uniform

a    b    c
d    X    e
f    g    h

In the picture above, the lower-case lettered points are known while X is the unknown/missing value.

There are a few heuristic solutions:

1. val(X) = average of vals are b,g (dvfs direction) and d, e (itr directions) or succinctly, X = (b + g + d + e) / 4.

2. linearly interpolate along each axis. on the dvfs axis (since it's uniformly spaced), this is the same as a simple
average i.e. X1 = (b + g) / 2

the itr axis is not uniform ([2, 10, 20, 30, 40, 50, 100, 200, 300, 400]), and if X.itr = 50, we shouldn't
average the values at itr=40 and itr=100 since they are not equidistant at all. instead, do linear interpolation

X2 = alpha * d + (1-alpha)*e where alpha = (X.itr - d.itr) / (e.itr - d.itr)

To get a final estimate, just average X1 and X2 i.e. X = (X1 + X2) / 2

3. one can also perform a full bilinear interpolation with the 4 corners a,c,f,h. such an interpolation doesn't
use b, d, e and g and instead

a. first compute b by doing a linear interpolation of a and c along the itr axis
b. compute g by doing a linear interpolation of f and h along the itr axis
c. compute X by doing a linear interpolation of the results from steps (a) and (b) along the dvfs direction

(btw. this also works if a and b are done along the dvfs direction and c is done along the itr direction)

Note that the estimates of b, d, e and g will not necessarily agree with the values measured at those points

Also, note that the assumption is that there aren't missing points that are adjacent to each other (which doesn't
seem to be the case)

check_simple_interp implements 1

check_linear_interp implements 2

'''
    
def check_linear_interp(target_vals, nb_vals):
    '''Quick and (very) dirty version. Sorry!
    '''
    
    assert(len(target_vals)==len(nb_vals))
    
    #t = (itr, dvfs, metric val)
    
    actuals = []
    preds = [] #[np.mean(list(n.values())) for n in nb_vals]
    
    for idx, n in enumerate(nb_vals): #n = {(itr+/-1, dvfs): val, (itr, dvfs+/-1):val}
        if len(n) != 4: #ignoring points on boundaries (includes corners)
            continue
                
        target_itr, target_dvfs, target_metric_val = target_vals[idx] #target value
                
        r = {'const_itr': [], 'const_dvfs': []} #split 4 points two sets
        for nb in n:
            itr = nb[0]
            dvfs = nb[1]
            
            if itr==target_itr:
                r['const_itr'].append((dvfs, n[nb]))
            
            if dvfs==target_dvfs:
                r['const_dvfs'].append((itr, n[nb]))
                
        pred = []
        for k in r:
            vals = r[k]
            
            idx_list = np.argsort([v[0] for v in vals])
            keys = np.array([v[0] for v in vals])[idx_list]
            metric_list = np.array([v[1] for v in vals])[idx_list]
            
            if k=='const_itr':
                alpha = (target_dvfs - keys[0]) / (keys[1] - keys[0])
                assert(alpha==0.5)
                pred.append(alpha*metric_list[0] + (1-alpha)*metric_list[1])
                                
            elif k=='const_dvfs':
                alpha = (target_itr - keys[0]) / (keys[1] - keys[0])
                
                pred.append(alpha*metric_list[0] + (1-alpha)*metric_list[1])
                
            else:
                raise ValueError(f'unknown key = {k} found')
        preds.append(np.mean(pred))
        actuals.append(target_metric_val)
        
    plt.figure()
    plt.plot(preds, actuals, 'p')
    plt.xlabel('preds')
    plt.ylabel('actuals')
    plt.plot(preds, preds) #diagonal
    
    error = np.sqrt(np.mean((np.array(actuals)/np.array(preds) - 1)**2))
    plt.title(f'Linear Interp of nbs: RMS of percentage error = {100*error:.3f}%')
        
def fill_missing_row(df, row):
    idx_names = df.index.names
    
    df.reset_index(inplace=True)
    df.loc[df.shape[0]] = row
    df.set_index(idx_names, inplace=True)
    
    return df

def plot_missing(df, tag):
    d = df[2]
    df = df[0]
    
    missing = []
    
    plt.figure()
    for i in d['itr']:
        for j in d['dvfs']:
            try:
                df.loc[i,j]['joules_mean']
                plt.plot([i], [j], 'o', color='g')
            except:
                missing.append([i,j])
                plt.plot([i], [j], 'x', color='r')
                
    plt.title(tag)
    
    return missing
    

def fill_missing_df(df, metric, debug=False):
    uniq_val_dict = df[2]
    itr_nbs = get_nbs(uniq_val_dict['itr'])
    dvfs_nbs = get_nbs(uniq_val_dict['dvfs'])

    missing_list = missing_keys(df) #get missing keys
    print(missing_list)
    
    imputed_vals = []
    
    if debug: print(missing_list)
    for miss in missing_list:
        target_itr = miss[0]
        target_dvfs = miss[1]
        

        #find neighbors of (itr, dvfs) -> (itr+/-1, dvfs), (itr, dvfs+/-1)
        val_dict = {}
        
        if debug: print(f'Current missing key = {miss}')
        for itr_n in itr_nbs[target_itr]:
            if itr_n is not None:
                if debug: print(itr_n, target_dvfs)
                try:
                    val = df_lookup.loc[itr_n, target_dvfs][metric]
                    val_dict[(itr_n, target_dvfs)] = val
                except:
                    if debug: print(f'Key = {(itr_n, target_dvfs)} missing')

        for dvfs_n in dvfs_nbs[target_dvfs]:
            if dvfs_n is not None:
                if debug: print(target_itr, dvfs_n)
                try:
                    val = df_lookup.loc[target_itr, dvfs_n][metric]
                    val_dict[(target_itr, dvfs_n)] = val
                except:
                    if debug: print(f'Key = {(target_itr, dvfs_n)} missing')
        
        if debug: print(val_dict)
    
        r = {'const_itr': [], 'const_dvfs': []} #split 4 points two sets
        for nb in val_dict:
            itr = nb[0]
            dvfs = nb[1]
            
            if itr==target_itr:
                r['const_itr'].append((dvfs, val_dict[nb]))
            
            if dvfs==target_dvfs:
                r['const_dvfs'].append((itr, val_dict[nb]))
        if debug: print(r)
        
        pred = []
        for k in r:
            vals = r[k]
            if len(vals)<2: continue
            
            idx_list = np.argsort([v[0] for v in vals])
            keys = np.array([v[0] for v in vals])[idx_list]
            metric_list = np.array([v[1] for v in vals])[idx_list]
            
            if k=='const_itr':
                alpha = (target_dvfs - keys[0]) / (keys[1] - keys[0])
                assert(alpha==0.5)
                pred.append(alpha*metric_list[0] + (1-alpha)*metric_list[1])
                                
            elif k=='const_dvfs':
                alpha = (target_itr - keys[0]) / (keys[1] - keys[0])
                
                pred.append(alpha*metric_list[0] + (1-alpha)*metric_list[1])
                
            else:
                raise ValueError(f'unknown key = {k} found')
        if debug: print(f'means across axes: {pred}')
            
        if len(pred)>0:
            imp_val = np.mean(pred)
        else:
            imp_val = 999
        
        imputed_vals.append(imp_val)
        if debug: print('----')
            
    assert(len(imputed_vals)==len(missing_list))
    
    return imputed_vals

def perform_bayesopt(metric = 'read_99th_mean',
                     minimize = True,
                     fill_missing = True
                     ):
    
    results = {}
    counter = 0
    for df_key in df_dict:
        print(df_key)
        for var_type in ['discrete']:#, 'range']:
            df, index_cols, uniq_vals = df_dict[df_key]

            if fill_missing:
                missing_val = df[metric].max()#want to minimize latency so choose high number
            
            if minimize:
                true_optimum = df[metric].min()
            else:
                true_optimum = df[metric].max()
            true_optimum_pt = df[df[metric]==true_optimum].index[0]
            
            search_space = prepare_search_space(df, var_type=var_type)
            pprint(search_space)
                            
            best_params, values, exp, model = optimize(parameters=search_space,
                                                       evaluation_function=lambda params: mcd_eval_func(params, df, index_cols, uniq_vals, metric, fill_missing=fill_missing, missing_val=missing_val),
                                                       experiment_name=f'{df_key}_{var_type}',
                                                       objective_name='mcd',
                                                       minimize=minimize,
                                                       total_trials=30,
                                                      )
            
            results[df_key, var_type] = (best_params, values, exp, model, minimize, true_optimum, true_optimum_pt, metric)
            counter += 1    

    return results

def plot_traces(results):
    key = list(results.keys())[0]
    for key in results:
        best_params, values, exp, model, minimize, true_optimum, true_optimum_pt, metric = results[key]
        if minimize:
            func_acc = np.minimum.accumulate
            func = np.minimum
            title = f'Minimum objective so far - {key[0]}'
        else:
            func_acc = np.maximum.accumulate
            func = np.maximum
            title = f'Maximum objective so far - {key[0]}'

        #render(plot_contour(model=model, param_x="itr", param_y='dvfs', metric_name='mcd'))

        obj_vals = np.array([[trial.objective_mean for trial in exp.trials.values()]])

        trace_plot = optimization_trace_single_method(y=func_acc(obj_vals, axis=1),
                                                      optimum=true_optimum,
                                                      title=title,
                                                      ylabel=metric)

        render(trace_plot)

        trace_plot = optimization_trace_single_method(y=obj_vals,
                                                      optimum=true_optimum,
                                                      title=title,
                                                      ylabel=metric)

        render(trace_plot)    

def plot_results(df_dict, key, results):
    df_lookup, _, uniq_val_dict = df_dict[key[0]]
    
    best_params, values, exp, model, minimize, true_optimum, true_optimum_pt, metric = results[key]
    
    #opacity
    alpha = df_lookup['read_99th_mean'].copy()
    #alpha = 1. / alpha
    alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min())
    alpha = alpha.to_dict()
    
    plt.figure()
    #raw grid with missing points
    counter = 0
    for itr in uniq_val_dict['itr']:
        for dvfs in uniq_val_dict['dvfs']:
            try:
                df_lookup.loc[itr, dvfs]
                a = alpha.get((itr, dvfs), 0.1)
                plt.plot([itr], [dvfs], 'o', c='g', alpha=a)
                
            except:
                if counter == 0:
                    plt.plot([itr], [dvfs], 'D', c='orange', label='Missing data')
                else:
                    plt.plot([itr], [dvfs], 'D', c='orange')
                counter += 1
                
        
    for idx, obs in enumerate(model.get_training_data()):
        pt = obs.features.parameters
        
        #don't plot the best param here
        if pt['itr']==best_params['itr'] and pt['dvfs']==best_params['dvfs']: continue

        if idx==0:
            plt.plot([pt['itr']], [pt['dvfs']], 'x', c='r', markersize=10, label='BayesOpt Sample')
        else:
            plt.plot([pt['itr']], [pt['dvfs']], 'x', c='r', markersize=10)
    
    plt.plot([true_optimum_pt[0]], [true_optimum_pt[1]], '+', c='r', label='True Optimum', markersize=20)
    
    plt.plot([best_params['itr']], [best_params['dvfs']], '*', c='r', label='BayesOpt Solution', markersize=12)
    
    best_val = values[0]['mcd']
    pct_diff = (best_val - true_optimum) / true_optimum * 100
    
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.title(f"OS={key[0][0].split('_')[0]} - Memcached QPS = {key[0][1]}\nTrue Optimum = {best_val:.2f} \nBest BayesOpt = {true_optimum:.2f}")
    
    plt.xlabel('ITR ($\mu$s)')
    plt.ylabel('DVFS')
    
    print(true_optimum_pt, best_params)
    return df_lookup

if __name__=='__main__':
    df_dict = prepare_data()

    results = perform_bayesopt(metric = 'read_99th_mean',
                               minimize = True,
                               fill_missing = True
                               )