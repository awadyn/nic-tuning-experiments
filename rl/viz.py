import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import utils

#start time, end time filtering
#add entropy of normalized entropy
#percentiles, bin counts

def prepare_data(loc):
    df = utils.combine_data(loc)
    df.set_index(['fname', 'sys'], inplace=True)

    for col in ['itr', 'rapl']:
        df[col] = df[col].astype(int)

    df['dvfs'] = df['dvfs'].apply(lambda x: int(x, base=16))

    df = df.sort_values(by=['itr', 'dvfs', 'rapl'])

    cols = [j for j in np.unique(["_".join(i.split('_')[:-1:]) for i in df.columns]) if j.find('per')==-1]

    #for c in df.columns:


    '''
    Normalizations:

    Heatmap:
    

    Correlation:
    (N, D) x (D, N) -> (N, N) - sort by cols (hover etc.)

    '''

def normalize(df, cols, pcts):
    df_list = []

    #percentile columns
    for c in cols:
        min_val = df[[f'{c}_{p}' for p in pcts]].min().min()
        max_val = df[[f'{c}_{p}' for p in pcts]].max().max()

        temp = (df[[f'{c}_{p}' for p in pcts]] - min_val) / (max_val - min_val)

        df_list.append(temp)

    df_norm = pd.concat(df_list, axis=1)
    df_norm.drop([f'c6_{p}' for p in pcts], axis=1, inplace=True)

    #static columns
    static_cols = set(df.columns).difference([f'{c}_{p}' for c in cols for p in pcts])
    df_static = df[static_cols].copy()

    for c in ['itr', 'dvfs', 'rapl', 'time_per_interrupt', 'joules_per_interrupt', 'read_99th']:
        df_static[c] = (df_static[c] - df_static[c].min()) / (df_static[c].max() - df_static[c].min())

    df_static.drop([f'read_{p}' for p in ['5th', '10th', '50th', '90th', '95th', 'std', 'min', 'avg']], axis=1, inplace=True)
    df_static.drop('rapl', axis=1, inplace=True)

    return df_norm, df_static, pd.concat([df_norm, df_static], axis=1)

#pct = np.unique([i.split('_')[-1] for i in df2.columns])[:-1:]

#TODO: keep X_1, X_99 and normalized dist
'''
df_list = []
for c in cols:
    select_cols = [f'{c}_{p}' for p in pct]
    print(select_cols)
    df_list.append(df[select_cols].apply(lambda x: x / x.sum(), axis=1))
'''