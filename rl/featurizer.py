import re
import os
from os import path
import sys
import time
import numpy as np
import pandas as pd

LINUX_COLS = ['i', 
              'rx_desc', 
              'rx_bytes', 
              'tx_desc', 
              'tx_bytes', 
              'instructions', 
              'cycles', 
              'ref_cycles', 
              'llc_miss', 
              'c1', 
              'c1e', 
              'c3', 
              'c6', 
              'c7', 
              'joules', 
              'timestamp']
TIME_CONVERSION_khz = 1./(2899999*1000)
JOULE_CONVERSION = 0.00001526

'''
Might need files from all cores for some metrics
Features need to be carefully chosen unlike here
'''

def parse_logs_to_df(fname):
    df = pd.read_csv(fname, sep=' ', names=LINUX_COLS)
    #df = df[(df['timestamp'] >= START_RDTSC) & (df['timestamp'] <= END_RDTSC)]

    df_non0j = df[(df['joules']>0) & (df['instructions'] > 0) & (df['cycles'] > 0) & (df['ref_cycles'] > 0) & (df['llc_miss'] > 0)].copy()
    df_non0j['timestamp'] = df_non0j['timestamp'] - df_non0j['timestamp'].min()
    df_non0j['timestamp'] = df_non0j['timestamp'] * TIME_CONVERSION_khz
    df_non0j['joules'] = df_non0j['joules'] * JOULE_CONVERSION

    tmp = df_non0j[['instructions', 'cycles', 'ref_cycles', 'llc_miss', 'joules', 'c1', 'c1e', 'c3', 'c6', 'c7']].diff()
    tmp.columns = [f'{c}_diff' for c in tmp.columns]
    df_non0j = pd.concat([df_non0j, tmp], axis=1)
    df_non0j.dropna(inplace=True)
    df.dropna(inplace=True)
    df_non0j = df_non0j[df_non0j['joules_diff'] > 0]

    return df, df_non0j, fname

def compute_features(df, df_non0j, fname):
    '''Features for each log file
    '''
    
    #chunk the file -> K interrupts
    #tx_bytes -> compute_entropy
    #tx_bytes -> 10, 25, 50, 75, 90 percentiles ...
    #percentiles, standard deviation, entropy

    #intuition: describe the "true" state of the OS/system

    #TODO: RDTSC
    #TODO: LATENCY <- energy, time, latency numbers
    
    percentile_list = [1, 10, 25, 50, 75, 90, 99] #MOVE TO CONFIG

    features = {'fname': fname}
    for col in ['instructions',
                'cycles',
                'ref_cycles',
                'llc_miss',
                'c1',
                'c1e',
                'c3',
                'c6',
                'c7',
                'joules']:

        pcs = np.percentile(df_non0j[col], percentile_list)

        for i in range(len(percentile_list)):
            features[f'{col}_{percentile_list[i]}'] = pcs[i]

    for col in ['rx_desc',
                'rx_bytes',
                'tx_desc',
                'tx_bytes']:

        pcs = np.percentile(df[col], percentile_list)

        for i in range(len(percentile_list)):
            features[f'{col}_{percentile_list[i]}'] = pcs[i]

    features['joules_per_interrupt'] = df_non0j['joules'].sum() / df.shape[0]
    features['time_per_interrupt'] = df_non0j['timestamp'].max() / df.shape[0]

    return features

def compute_entropy(df, colname, nonzero=False):
    x  = df[colname].value_counts()

    if nonzero:
        x = x[x.index > 0]

    x = x / x.sum() #scipy.stats.entropy actually automatically normalizes

    return entropy(x)

if __name__=='__main__':
    if len(sys.argv) != 3:
        raise ValueError("Usage: python featurizer.py [dmesg log] [loc]")
    
    fname = sys.argv[1]
    loc = sys.argv[2]
    outfile = loc + '/' + fname.split('/')[-1] + '_features.csv'

    df, df_non0j, fname = parse_logs_to_df(fname) #log file -> dataframe
    features = compute_features(df, df_non0j, fname) #dataframe -> vector of features

    pd.DataFrame(features, index=[0]).to_csv(outfile, index=False)