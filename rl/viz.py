import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import torch
import utils
import config

#start time, end time filtering
#add entropy of normalized entropy
#percentiles, bin counts

plt.ion()

def plot_policy(policy, env, itr_list, dvfs_list):
    pred_dict = {}
    plt.figure()
    N_outputs_per_knob = config.N_outputs_per_knob

    for itr in itr_list:
        for dvfs in dvfs_list:
            plt.plot([itr], [dvfs], marker="o", c='b', markersize=2)

    for itr in itr_list:
        for dvfs in dvfs_list:
            pred = policy(torch.tensor(list(env.state_dict[(itr, dvfs, 135, 'linux', 0, 0)].values())).unsqueeze(0)).detach().numpy()

            pred_dict[(itr, dvfs)] = pred

            #plt.plot([itr], [dvfs], marker="o", c='b', markersize=1)

            target_point = []
            for idx in np.arange(int(pred.shape[1]/N_outputs_per_knob)):
                target = pred[0, (N_outputs_per_knob*idx):(N_outputs_per_knob*idx+N_outputs_per_knob)].argmax()
                
                if N_outputs_per_knob==2:
                    target = {0: -1, 1:1}[target]
                elif N_outputs_per_knob==3:
                    target = {0: -1, 1:0, 2:1}[target]

                colname = env.col_list[idx]
                if colname=='itr':
                    target = env.action_space[env.col_list[idx]][itr][target]
                elif colname=='dvfs':
                    target = env.action_space[env.col_list[idx]][dvfs][target]

                target_point.append(target)

            plt.arrow(itr, dvfs, target_point[0]-itr, target_point[1]-dvfs, width=0.001, head_width=10, head_length=100, alpha=0.5, length_includes_head=True)

            print(itr, dvfs, target_point)

    plt.xlabel('itr')
    plt.ylabel('dvfs')

    return pred_dict

'''
def prepare_data(loc, save_loc=None):
    df = utils.combine_data(loc)
    df.set_index(['fname', 'sys'], inplace=True)

    for col in ['itr', 'rapl']:
        df[col] = df[col].astype(int)

    df['dvfs'] = df['dvfs'].apply(lambda x: int(x, base=16) if type(x) is str else x) #need to improve

    df = df.sort_values(by=['itr', 'dvfs', 'rapl'])

    if 'level_0' in df:
        df.drop('level_0', axis=1, inplace=True)

    percentile_cols = np.unique(['_'.join(c.split('_')[:-1:]) for c in df.columns if c.find('_')>-1 and c.find('read')==-1 and c.find('_per_')==-1])
    percentile_vals = [1, 10, 25, 50, 75, 90, 99] #MOVE TO CONFIG

    if save_loc:
        df.to_csv(f'{loc}/features.csv', index=False)

    return df, percentile_cols, percentile_vals

def normalize(df, cols, pcts, save_loc=None):
    df_list = []

    #percentile columns
    for c in cols:
        df_block = df[[f'{c}_{p}' for p in pcts]] 

        min_val = df_block.min().min()
        max_val = df_block.max().max()

        if max_val == min_val:
            continue

        temp = (df_block - min_val) / (max_val - min_val)

        df_list.append(temp)

    df_norm = pd.concat(df_list, axis=1)

    #static columns
    static_cols = set(df.columns).difference([f'{c}_{p}' for c in cols for p in pcts])
    df_static = df[static_cols].copy()

    #normalize static cols for plotting
    for c in ['itr', 'dvfs', 'rapl', 'time_per_interrupt', 'joules_per_interrupt', 'read_99th']:
        df_static[c] = (df_static[c] - df_static[c].min()) / (df_static[c].max() - df_static[c].min())
    df_static.drop([f'read_{p}' for p in ['5th', '10th', '50th', '90th', '95th', 'std', 'min', 'avg']], axis=1, inplace=True)
    df_static.drop('rapl', axis=1, inplace=True)

    df_plot = pd.concat([df_norm, df_static], axis=1)
    assert(df_plot.shape[0]==df_norm.shape[0]==df_static.shape[0])
    
    if save_loc:
        df_plot.to_csv(f'{save_loc}/features_norm.csv', index=True)

    return df_norm, df_static, df_plot

def plot_df(df, row_start, row_end):
    plt.figure()

    plt.imshow(df.iloc[row_start:row_end+1])

'''