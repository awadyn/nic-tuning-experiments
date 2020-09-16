import pandas as pd
import numpy as np
import os
import glob

'''
TODO: 
1. make process_data flexible for non-netpipe data
msg -> qps (memcached) or None (nodejs)
'''

def scan_all_data(name, column_names=None, TIME_CONVERSION=1, JOULE_CONVERSION=1, N_THREADS=1):
    '''Detailed logs -> summarized data
    '''

    filelist = glob.glob(name)

    def process_data(filelist, dlist):
        d = {
         'sys': [],
         'rnd': [],
         'msg': [],
         'itr': [],
         'dvfs': [],
         'rapl': [],
         'joules': [],
         'time': [],
         'edp': [],
         'llcmiss_total': [],
         'cycles_total': [],
         'ins_total': [],
         }

        for idx, f in enumerate(filelist):
            if idx % 1000 == 0: print(idx)

            #if idx==2: break

            line_split = f.replace('.csv','').split('.')

            system = line_split[0].split('/')[-1]
            param_split = line_split[2].split('_')

            rnd = param_split[0]
            msg = param_split[1]
            itr = param_split[3]
            dvfs = param_split[4]
            rapl = param_split[5]

            #print(f)
            df = pd.read_csv(f, sep = ' ', names=column_names)

            #cleanup
            df_non0j = df[df['joules']>0].copy()

            drop_cols = [c for c in df_non0j.columns if c.find('Unnamed')>-1]
            if len(drop_cols) > 0: df_non0j.drop(drop_cols, axis=1, inplace=True)
            #print(f'Dropping null rows: {df_non0j.shape[0] - df_non0j.dropna().shape[0]} rows')
            df_non0j.dropna(inplace=True)
        
            #reset 0 for metrics and convert units
            df_non0j['timestamp'] = df_non0j['timestamp'] - df_non0j['timestamp'].min()
            df_non0j['joules'] = df_non0j['joules'] - df_non0j['joules'].min()
            
            df_non0j['timestamp'] = df_non0j['timestamp'] * TIME_CONVERSION
            df_non0j['joules'] = df_non0j['joules'] * JOULE_CONVERSION

            assert(df_non0j.shape==df_non0j.dropna().shape)
        
            #get edp value
            last_row = df_non0j.tail(1).iloc[0]
            edp_val = 0.5 * last_row['joules'] * last_row['timestamp']
            
            d['sys'].append(system)
            d['rnd'].append(rnd)
            d['msg'].append(msg)
            d['itr'].append(itr)
            d['dvfs'].append(dvfs)
            d['rapl'].append(rapl)

            d['joules'].append(last_row['joules'])
            d['time'].append(last_row['timestamp'])
            d['edp'].append(edp_val)
        
            d['llcmiss_total'].append(df_non0j["llc_miss"].sum())
            d['ins_total'].append(df_non0j["instructions"].sum())
            d['cycles_total'].append(df_non0j["cycles"].sum())
            
        return pd.DataFrame(d)
    
    return process_data(filelist, [])

def start_netpipe_analysis(drop_outliers=False, old=False):
    N_ROUNDS = 5000

    if old:
        df_fixed = pd.read_csv('jul20data.csv')
    else:
        df_fixed = pd.read_csv('aug11data.csv') #truncated logs for 512k fixed
    df_gov = pd.read_csv('jul20data_governor.csv')

    df_fixed = rename_cols(df_fixed)
    df_gov = rename_cols(df_gov)

    #outlier_list = identify_outliers(df, dfr)
    #df_fixed = filter_outliers(df_fixed)

    df_fixed['gov'] = 0
    df_gov['gov'] = 1

    df_fixed = convert_units(df_fixed)
    df_gov = convert_units(df_gov)

    df = pd.concat([df_fixed, df_gov], axis=0)
    df['tput'] = df['msg']*N_ROUNDS / df['time']

    df['eput'] = df['msg']*N_ROUNDS / df['joules']

    df_comb = prepare_scan_all_data(df)
    df_comb.reset_index(inplace=True)

    outlier_list = identify_outliers(df_comb, df)
    if drop_outliers:    
        print(f'Dropping {len(outlier_list)} outlier rows')

        print(f'Before: {df.shape[0]}')
        df = filter_outliers(outlier_list, df)
        print(f'After: {df.shape[0]}')

        df_comb = prepare_scan_all_data(df)

    return df_comb, df, outlier_list

def start_nodejs_analysis(filename, drop_outliers=False, scale_requests=False):
    df = pd.read_csv(filename, sep=' ')

    df.drop(['START_RDTSC', 'END_RDTSC'], axis=1, inplace=True)

    df = rename_cols(df)

    print('Dropping rows with time <= 29')
    print(f'Before: {df.shape[0]}')
    df = df[df['time']>29].copy()
    print(f'After: {df.shape[0]}\n')

    def scale_to_requests(d):
        COLS_TO_SCALE = ['requests',
                         'time',
                         'joules',
                         'instructions',
                         'cycles',
                         'refcyc',
                         'llc_miss',
                         'c3',
                         'c6',
                         'c7']
        SCALE_FACTOR = 100000. / d['requests']

        for c in COLS_TO_SCALE:
            d[c] = d[c] * SCALE_FACTOR

        return d

    if scale_requests:
        df = scale_to_requests(df)

    df['edp'] = 0.5 * (df['joules'] * df['time'])
    df['tput'] = df['requests'] / df['time']
    df['eput'] = df['requests'] / df['joules']

    dfr = df.copy() #raw data
    df = prepare_scan_all_data(dfr) #grouped by data
    df.reset_index(inplace=True)

    outlier_list = identify_outliers(df, dfr)
    if drop_outliers:    
        print(f'Dropping {len(outlier_list)} outlier rows')

        print(f'Before: {dfr.shape[0]}')
        dfr = filter_outliers(outlier_list, dfr)
        print(f'After: {dfr.shape[0]}')

        df = prepare_scan_all_data(dfr) #grouped by data    

        df.reset_index(inplace=True)

    return df, dfr, outlier_list

def start_mcd_analysis(filename, drop_outliers=False, scale_requests=True):
    '''TODO: Merge with start_nodejs_analysis'''
    df = pd.read_csv(filename, sep=' ')

    df = rename_cols(df)

    #fix QPS
    possible_qps_vals = np.array([200000, 400000, 600000])
    print(f"Possible QPS values : {possible_qps_vals}")
    def cluster_qps_values(df):
        df['QPS_uncorrected'] = df['QPS']
        
        df['QPS'] = df['QPS'].apply(lambda x: possible_qps_vals[np.argmin(np.abs((x - possible_qps_vals)))])

        return df

    df = cluster_qps_values(df)

    #drop time < 30s
    print('Dropping rows with time <= 29')
    print(f'Before: {df.shape[0]}')
    df = df[df['time']>29].copy()
    print(f'After: {df.shape[0]}\n')

    #drop 99th percentile > 500 latencies
    print('Dropping rows with read_99th latency > 500')
    print(f'Before: {df.shape[0]}')
    df = df[df['read_99th'] <= 500].copy()
    print(f'After: {df.shape[0]}\n')

    def scale_to_requests(d):
        print("Scaling to 5 million requests")
        COLS_TO_SCALE = ['time',
                         'joules',
                         'instructions',
                         'cycles',
                         'refcyc',
                         'llc_miss',
                         'c3',
                         'c6',
                         'c7']

        SCALE_FACTOR = 5000000. / (d['QPS_uncorrected']*d['time'])

        for c in COLS_TO_SCALE:
            d[c] = d[c] * SCALE_FACTOR

        return d

    if scale_requests:
        df = scale_to_requests(df)


    df['edp'] = 0.5 * (df['joules'] * df['time'])
    df['eput'] = df['QPS'] * df['time'] / df['joules']
    #df['tput'] = df['requests'] / df['time']
    #df['eput'] = df['requests'] / df['joules']

    dfr = df.copy() #raw data
    df = prepare_scan_all_data(dfr) #grouped by data
    df.reset_index(inplace=True)

    outlier_list = identify_outliers(df, dfr)
    if drop_outliers:    
        print(f'Dropping {len(outlier_list)} outlier rows')

        print(f'Before: {dfr.shape[0]}')
        dfr = filter_outliers(outlier_list, dfr)
        print(f'After: {dfr.shape[0]}')

        df = prepare_scan_all_data(dfr) #grouped by data    
        df.reset_index(inplace=True)

    return df, dfr, outlier_list


def start_mcdsilo_analysis(filename, drop_outliers=False, scale_requests=False):
    df = pd.read_csv('aug19_mcdsilologs/mcdsilo_combined.csv', sep=' ')

    df = rename_cols(df)

    possible_qps_vals = np.array([50000, 100000, 200000])
    print(f"Possible QPS values : {possible_qps_vals}")
    def cluster_qps_values(df):
        df['QPS_uncorrected'] = df['QPS']
        
        df['QPS'] = df['QPS'].apply(lambda x: possible_qps_vals[np.argmin(np.abs((x - possible_qps_vals)))])

        return df

    df = cluster_qps_values(df)

    #drop time < 30s
    print('Dropping rows with time <= 19')
    print(f'Before: {df.shape[0]}')
    df = df[df['time']>19].copy()
    print(f'After: {df.shape[0]}\n')

    #drop 99th percentile > 500 latencies
    print('Dropping rows with read_99th latency > 500')
    print(f'Before: {df.shape[0]}')
    df = df[df['read_99th'] <= 500].copy()
    print(f'After: {df.shape[0]}\n')

    def scale_to_requests(d):
        print("Scaling to 1 million requests")
        COLS_TO_SCALE = ['time',
                         'joules',
                         'instructions',
                         'cycles',
                         'refcyc',
                         'llc_miss',
                         #'c3',
                         #'c6',
                         #'c7'
                         ]

        SCALE_FACTOR = 1000000. / (d['QPS_uncorrected']*d['time'])

        for c in COLS_TO_SCALE:
            try:
                d[c] = d[c] * SCALE_FACTOR
            except:
                print("problem with column", c)                

        return d

    if scale_requests:
        df = scale_to_requests(df)

    df['edp'] = 0.5 * (df['joules'] * df['time'])
    #df['tput'] = df['requests'] / df['time']
    #df['eput'] = df['requests'] / df['joules']

    dfr = df.copy() #raw data
    df = prepare_scan_all_data(dfr) #grouped by data
    df.reset_index(inplace=True)

    outlier_list = identify_outliers(df, dfr)
    if drop_outliers:    
        print(f'Dropping {len(outlier_list)} outlier rows')

        print(f'Before: {dfr.shape[0]}')
        dfr = filter_outliers(outlier_list, dfr)
        print(f'After: {dfr.shape[0]}')

        df = prepare_scan_all_data(dfr) #grouped by data    
        df.reset_index(inplace=True)

    return df, dfr, outlier_list
