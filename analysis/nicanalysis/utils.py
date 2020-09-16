import pandas as pd

def convert_units(df):
    '''Convert raw counters into physical units
    '''

    df['joules'] = df['joules'] * JOULE_CONVERSION
    df['edp'] = df['edp'] * JOULE_CONVERSION

    df['time'] = df['time'] * TIME_CONVERSION_khz
    df['edp'] = df['edp'] * TIME_CONVERSION_khz

    return df

def identify_outliers(df, dfr, RATIO_THRESH=0.03):
    '''Each configuration (itr, rapl, dvfs, msg/qps/None) has multiple runs.
    Some runs might have erroneous data that should be discarded.
    This function identifies the outliers with a simple rule of thumb.

    df -> grouped by data 
    dfr -> data for individual runs
    '''

    df = df.copy()

    df.fillna(0, inplace=True) #configs with 1 run have std dev = 0

    #Criterion for removing outliers
    df_highstd = df[df.joules_std / df.joules_mean > RATIO_THRESH] #step 1

    outlier_list = []

    for idx, row in df_highstd.iterrows(): #loop over configs with high std dev / mean
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

        else: #nodejs
            df_bad = dfr[(dfr.sys==sys) & (dfr.itr==itr) & (dfr.dvfs==dvfs) & (dfr.rapl==rapl)]

            bad_row = df_bad[df_bad.joules==df_bad.joules.min()].iloc[0] #the outlier doesn't have to be the minimum joule one

            outlier_list.append((sys, bad_row['i'], itr, dvfs, rapl)) #can ignore "i" to focus on bad config

    return outlier_list

def filter_outliers(lines, dfr):
    '''Be careful about order of cols. Should match identify_outliers or pass dict
    TODO: Get COLS from identify_outliers and pass them as arg to prevent errors
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

