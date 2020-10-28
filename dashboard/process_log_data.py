import pandas as pd
import numpy as np
import operator
from config import *

'''
* Don't do filename processing here.
* All log files should be "uniformized" in preprocess.py
'''

def process_rdtsc_limits(ts_filename, start_idx=1, end_idx=2):
    '''Note: looking at every entry found in rdtsc file, not just the first one
    '''        
    with open(ts_filename) as f:
        lines = f.readlines()
        lines = [l for l in lines if len(l)>0]
        
        time_intervals = []
        for l in lines:
            l_split = l.split()
            start, end = int(l_split[start_idx]), int(l_split[end_idx])

            tval = np.abs(end-start) * TIME_CONVERSION_khz
            if np.abs(tval - 30) < 0.2:
                print(f'Found time interval: {tval}')
                time_intervals.append((start, end, tval))

        if len(time_intervals)==0:
            raise ValueError("Didn't find valid time interval")
        if len(time_intervals)>1:
            raise ValueError("Found multiple time intervals")


        return time_intervals[0]

#TODO: fix ref_cycles for netpipe data
def process_log_file(filename, ts_filename=None, ts_start_idx=0, ts_end_idx=1, pass_colnames=True, skiprows=0):
    #read log file
    #TODO: remove infer_colnames, skiprows after making log files uniform
    if pass_colnames:
        df = pd.read_csv(filename, sep = ' ', names=COLS, skiprows=skiprows)
    else:
        df = pd.read_csv(filename, sep = ' ', skiprows=skiprows)

    df.columns = [COL_MAPPER.get(c, c) for c in df.columns]

    if ts_filename:
        start, end, interval_len_sec = process_rdtsc_limits(ts_filename, 
                                                            start_idx=ts_start_idx, 
                                                            end_idx=ts_end_idx)

    #COLS_TO_DIFF = ['instructions', 'cycles', 'ref_cycles', 'llc_miss', 'joules', 'timestamp']
    COLS_TO_DIFF = ['instructions', 'cycles', 'llc_miss', 'joules', 'timestamp']

    #----------
    #every interrupt
    df_orig = df.copy()

    for c in COLS_TO_DIFF: 
        df_orig[c] = df_orig[c] - df_orig[c].min()

    df_orig['timestamp'] = df_orig['timestamp'] * TIME_CONVERSION_khz
    df_orig['joules'] = df_orig['joules'] * JOULE_CONVERSION
    #---------

    #----------
    #only non-zero energy counters
    df = df[df['joules'] > 0].copy()

    for c in COLS_TO_DIFF: 
        df[c] = df[c] - df[c].min()

    df['timestamp'] = df['timestamp'] * TIME_CONVERSION_khz
    df['joules'] = df['joules'] * JOULE_CONVERSION

    #add diffs
    df2 = df[COLS_TO_DIFF + ['i']].diff()
    df2.columns = [f'{c}_diff' for c in df2.columns]

    df = pd.concat([df, df2], axis=1)
    #----------

    #compute busy time
    #df['nonidle_timestamp_diff'] = df['ref_cycles_diff'] * TIME_CONVERSION_khz
    #df['nonidle_frac_diff'] = df['nonidle_timestamp_diff'] / df['timestamp_diff']
    
    return df, df_orig

# Everything below is going to be removed soon

def process_nodejs_logs(df, slice_middle=False):
    df_orig = df.copy()
    for c in ['instructions', 'refcyc', 'cycles', 'llc_miss', 'joules', 'timestamp']: 
        df_orig[c] = df_orig[c] - df_orig[c].min()
    #---------

    df = df[df['joules'] > 0].copy()

    for c in ['instructions', 'refcyc', 'cycles', 'llc_miss', 'joules', 'timestamp']: 
        df[c] = df[c] - df[c].min()

    df2 = df[['instructions', 'refcyc', 'cycles', 'joules', 'timestamp', 'i']].diff()
    df2.columns = [f'{c}_diff' for c in df2.columns]

    df = pd.concat([df, df2], axis=1)

    print("Assuming >= 1 second time")
    if slice_middle:
        df = df[(df.timestamp >= df.timestamp.max()*0.5-0.1) & (df.timestamp <= df.timestamp.max()*0.5+0.1)]

    for c in ['instructions', 'refcyc', 'cycles', 'llc_miss', 'joules', 'timestamp']: 
        df[c] = df[c] - df[c].min()

    df['nonidle_timestamp_diff'] = df['refcyc_diff'] * TIME_CONVERSION_khz
    df['nonidle_frac_diff'] = df['nonidle_timestamp_diff'] / df['timestamp_diff']

    #----------
    if slice_middle:
        df_orig = df_orig[(df_orig.timestamp >= df.timestamp.max()*0.5-0.1) & (df_orig.timestamp <= df.timestamp.max()*0.5+0.1)]
    #----------

    return df,df_orig

def read_nodej_logfile(filename):
    EBBRT_COLS = ['i',
                  'rxdesc',
                  'rxbytes',
                  'txdesc',
                  'txbytes', 
                  'ins', 
                  'cyc', 
                  'refcyc', 
                  'llcm', 
                  'C3', 
                  'C6', 
                  'C7', 
                  'JOULE', 
                  'TSC']

    if filename.find('node_dmesg')>-1:
        df = pd.read_csv(filename, sep=' ', skiprows=1)

        tsc_file = filename.replace('_dmesg', '_rdtsc').replace('.csv', '')

        print(f'Log file      : {filename}')
        print(f'Timestamp file: {tsc_file}')

        found_time_interval = False
        with open(tsc_file) as f:
            lines = f.readlines()
            lines = [l for l in lines if len(l)>0]
            if len(lines) != 3:
                raise ValueError(f'{filename} found !=3 tsc limits')
            
            for l in lines:
                l_split = l.split()
                start, end = int(l_split[1]), int(l_split[2])

                tval = np.abs(end-start) * TIME_CONVERSION_khz
                if np.abs(tval - 30) < 0.2:
                    print(f'Found time interval: {tval}')
                    found_time_interval = True
                    break

            if not found_time_interval:
                raise ValueError("Didn't find valid time interval")
            print(f'Using time interval: {tval}')

    elif filename.find('ebbrt_dmesg')>-1:
        df = pd.read_csv(filename, sep=' ', skiprows=1, names=EBBRT_COLS)

        #tsc file
        tags = filename.split('.')
        tags[0] = tags[0].replace('_dmesg', '_rdtsc')
        tag1_split = tags[1].split('_')
        tags[1] = '_'.join([tag1_split[0]] + tag1_split[2:])
        tsc_file = '.'.join(tags[0:-1])

        print(f'Log file      : {filename}')
        print(f'Timestamp file: {tsc_file}')

        with open(tsc_file) as f:
            lines = f.readlines()
            lines = [l for l in lines if len(l)>0]
            if len(lines) != 1:
                raise ValueError(f'{filename} found multiple tsc limits')
            start, end = [int(l) for l in lines[0].split()]
            print(start, end)

    df = rename_cols(df)

    print("Filtering on timestamps")
    print(f"Before filtering: {df.shape[0]}")
    df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    print(f"After filtering: {df.shape[0]}")

    df["timestamp"] = df["timestamp"] - df["timestamp"].min()
    df["joules"] = df["joules"] - df["joules"].min()

    df['timestamp'] = df['timestamp'] * TIME_CONVERSION_khz
    df['joules'] = df['joules'] * JOULE_CONVERSION

    return df

def asplos_log_plots(filename, folder, skiprows=None, slice_middle=True):
    if folder=='jul20':
        df = pd.read_csv(filename, sep = ' ', names=COLS, skiprows=skiprows)
    else:
        df = pd.read_csv(filename, sep = ' ', skiprows=skiprows)

    if folder=='jul20':
        COLS_TO_DIFF = ['instructions', 'cycles', 'llc_miss', 'joules', 'timestamp']
    else:
        COLS_TO_DIFF = ['instructions', 'cycles', 'ref_cycles', 'llc_miss', 'joules', 'timestamp']

    #----------
    df_orig = df.copy()
    for c in COLS_TO_DIFF: 
        df_orig[c] = df_orig[c] - df_orig[c].min()
    df_orig['timestamp'] = df_orig['timestamp'] * TIME_CONVERSION_khz
    df_orig['joules'] = df_orig['joules'] * JOULE_CONVERSION
    #---------

    df = df[df['joules'] > 0].copy()

    for c in COLS_TO_DIFF: 
        df[c] = df[c] - df[c].min()

    df['timestamp'] = df['timestamp'] * TIME_CONVERSION_khz
    df['joules'] = df['joules'] * JOULE_CONVERSION

    df2 = df[COLS_TO_DIFF + ['i']].diff()
    df2.columns = [f'{c}_diff' for c in df2.columns]

    df = pd.concat([df, df2], axis=1)

    print("Assuming >= 1 second time")
    if slice_middle:
        df = df[(df.timestamp >= df.timestamp.max()*0.5-0.1) & (df.timestamp <= df.timestamp.max()*0.5+0.1)]

    for c in COLS_TO_DIFF: 
        df[c] = df[c] - df[c].min()

    if folder=='jul20':
        df['nonidle_timestamp_diff'] = df['cycles_diff'] * TIME_CONVERSION_khz
    else:
        df['nonidle_timestamp_diff'] = df['ref_cycles_diff'] * TIME_CONVERSION_khz

    df['nonidle_frac_diff'] = df['nonidle_timestamp_diff'] / df['timestamp_diff']

    #----------
    if slice_middle:
        df_orig = df_orig[(df_orig.timestamp >= df.timestamp.max()*0.5-0.1) & (df_orig.timestamp <= df.timestamp.max()*0.5+0.1)]
    #----------
    
    return df, df_orig

