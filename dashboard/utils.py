from config import *
import pandas as pd

def rename_cols(df):
    cols = []
    for c in df.columns:
        if c.lower().find('joule')>-1:
            cols.append('joules')

        elif c.lower()=='rnd':
            cols.append('i')

        elif c=='RDTSC_START':
            cols.append(c.lower())

        elif c=='RDTSC_END':
            cols.append(c.lower())

        elif c.lower().find('tsc')>-1:
            cols.append('timestamp')

        elif c.find('ins')>-1:
            cols.append('instructions')

        elif c.find('llcm')>-1:
            cols.append('llc_miss')

        #elif c.find('refcyc'):
        #    cols.append('refcycles')

        elif c=='cyc':
            cols.append('cycles')

        else:
            cols.append(c)
    df.columns = cols

    return df

def convert_units(df):
    df['joules'] = df['joules'] * JOULE_CONVERSION
    df['edp'] = df['edp'] * JOULE_CONVERSION

    df['time'] = df['time'] * TIME_CONVERSION_khz
    df['edp'] = df['edp'] * TIME_CONVERSION_khz

    return df

def identify_outliers(df, dfr, RATIO_THRESH=0.03):
    '''df -> grouped by data 
    dfr -> data for individual runs
    '''

    df = df.copy()

    df.fillna(0, inplace=True) #configs with 1 run have std dev = 0

    df_highstd = df[df.joules_std / df.joules_mean > RATIO_THRESH] #step 1

    outlier_list = []

    for idx, row in df_highstd.iterrows():
        sys = row['sys']
        itr = row['itr']
        dvfs = row['dvfs']
        rapl = row['rapl']

        if 'msg' in row: #netpipe
            msg = row['msg']

            df_bad = dfr[(dfr.sys==sys) & (dfr.itr==itr) & (dfr.dvfs==dvfs) & (dfr.rapl==rapl) & (dfr.msg==msg)]

            bad_row = df_bad[df_bad.joules==df_bad.joules.min()].iloc[0] #the outlier doesn't have to be the minimum joule one

            outlier_list.append((sys, bad_row['i'], itr, dvfs, rapl, msg)) #can ignore "i" to focus on bad config

        elif 'QPS' in row: #memcached
            qps = row['QPS']

            df_bad = dfr[(dfr.sys==sys) & (dfr.itr==itr) & (dfr.dvfs==dvfs) & (dfr.rapl==rapl) & (dfr.QPS==qps)]

            bad_row = df_bad[df_bad.joules==df_bad.joules.min()].iloc[0] #the outlier doesn't have to be the minimum joule one

            outlier_list.append((sys, bad_row['i'], itr, dvfs, rapl, qps)) #can ignore "i" to focus on bad config

        else:
            df_bad = dfr[(dfr.sys==sys) & (dfr.itr==itr) & (dfr.dvfs==dvfs) & (dfr.rapl==rapl)]

            bad_row = df_bad[df_bad.joules==df_bad.joules.min()].iloc[0] #the outlier doesn't have to be the minimum joule one

            outlier_list.append((sys, bad_row['i'], itr, dvfs, rapl)) #can ignore "i" to focus on bad config

    return outlier_list


def filter_outliers(lines, dfr):
    '''Careful about order of cols. Should match identify_outliers or pass dict
    '''

    COLS = ['sys', 'i', 'itr', 'dvfs', 'rapl']
    
    if 'msg' in dfr.columns: #netpipe
        COLS.append('msg')

    elif 'QPS' in dfr.columns: #memcached
        COLS.append('QPS')

    dfr.set_index(COLS, inplace=True)

    size_before = dfr.shape[0]
    dfr = dfr.drop(lines, axis=0).reset_index()
    size_after = dfr.shape[0]

    assert(size_before - size_after == len(lines))

    return dfr

def prepare_scan_all_data(df):
    if 'msg' in df.columns: #netpipe
        COLS = ['sys', 'msg', 'itr', 'dvfs', 'rapl']
    elif 'QPS' in df.columns:
        COLS = ['sys', 'QPS', 'itr', 'dvfs', 'rapl']
    else: #nodejs
        COLS = ['sys', 'itr', 'dvfs', 'rapl']    

    df_mean = df.groupby(COLS).mean()
    df_std = df.groupby(COLS).std()
    
    df_mean.columns = [f'{c}_mean' for c in df_mean.columns]
    df_std.columns = [f'{c}_std' for c in df_std.columns]

    df_comb = pd.concat([df_mean, df_std], axis=1)

    return df_comb
