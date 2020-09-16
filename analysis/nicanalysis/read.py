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

def read_mcd_logfile(filename, nz=False):
    #filter time
    #tsc_filename = np.unique([mcd_get_tsc_filename(f) for f in filenames])
    tsc_filename = mcd_get_tsc_filename(filename)
    print(filename, tsc_filename)

    start, end = mcd_get_time_bounds(tsc_filename)

    #read
    if filename.find('ebbrt')>-1:
        COLS = [
                'i',
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
                'TSC'
        ]

        df = pd.read_csv(filename, sep=' ', skiprows=1, names=COLS)
    else:
        df = pd.read_csv(filename, sep=' ', skiprows=1)

    df = rename_cols(df)

    #drop first two columns
    DROP_COLS = [c for c in df.columns if c.find('[')>-1 or c.find(']')>-1]
    df.drop(DROP_COLS, axis=1, inplace=True)

    if nz:
        df = df[df['joules']>0].copy()

    if end == -1:
        df = df[(df['timestamp']>=start)]
    else:
        df = df[(df['timestamp']>=start) & (df['timestamp']<=end)]

    df.sort_values('timestamp', ascending=True, inplace=True)

    return df    

def mcd_read_logfile_combined(sys, 
                              run_id, 
                              itr,
                              dvfs,
                              rapl,
                              qps,
                              nz=False,
                              src_folder='aug19_mcdlogs',
                              N=8,
                              silo=False):

    if sys=='linux':
        tag = 'mcd'
        extension = ''

    elif sys=='ebbrt':
        tag = sys
        extension = '.csv'
    else:
        raise ValueError("Use valid tag")

    if silo:
        tag2 = 'silo'
        if sys=='ebbrt':
            tag2 = ''
    else:
        tag2 = ''

    df_even = [read_mcd_logfile(f'{src_folder}/{tag}{tag2}_dmesg.{run_id}_{2*c}_{itr}_{dvfs}_{rapl}_{qps}{extension}', nz=nz) for c in range(N)]
    if silo:
        if N < 8:
            df_odd = [read_mcd_logfile(f'{src_folder}/{tag}{tag2}_dmesg.{run_id}_{2*c+1}_{itr}_{dvfs}_{rapl}_{qps}{extension}', nz=nz) for c in range(N)]
        else:
            df_odd = [read_mcd_logfile(f'{src_folder}/{tag}{tag2}_dmesg.{run_id}_{2*c+1}_{itr}_{dvfs}_{rapl}_{qps}{extension}', nz=nz) for c in range(N-1)]
    else:
        df_odd = [read_mcd_logfile(f'{src_folder}/{tag}{tag2}_dmesg.{run_id}_{2*c+1}_{itr}_{dvfs}_{rapl}_{qps}{extension}', nz=nz) for c in range(N)]

    return df_even, df_odd    

def mcd_prepare_for_edp(df_even, df_odd, core_even, core_odd, simple=False):
    df1, df2 = df_even[core_even], df_odd[core_odd]

    df1 = df1.copy()
    df2 = df2.copy()

    for col in ['joules', 'timestamp']:
        df1[col] = df1[col] - df1[col].min()
        df2[col] = df2[col] - df2[col].min()

    #for d in [df1, df2]:
    #    d['joules'] = d['joules'] * JOULE_CONVERSION
    #    d['timestamp'] = d['timestamp'] * TIME_CONVERSION_khz


    #
    df1['timestamp'] *= TIME_CONVERSION_khz
    df1['joules'] *= JOULE_CONVERSION

    df2['timestamp'] *= TIME_CONVERSION_khz
    df2['joules'] *= JOULE_CONVERSION

    if simple:
        return df1, df2, None

    #combined edp
    df_comb = pd.merge(df1, df2, how='outer', on='timestamp').sort_values(by='timestamp', ascending=True)
    print(f'df1: {df1.shape[0]}')
    print(f'df2: {df2.shape[0]}')
    print(f'Total: {df1.shape[0] + df2.shape[0]}')
    print(f'df_comb: {df_comb.shape[0]}')

    #df_comb.fillna(0, inplace=True)
    df_comb.fillna(method='ffill', inplace=True)
    df_comb['joules'] = df_comb['joules_x'] + df_comb['joules_y']
    #df_comb['timestamp'] *= TIME_CONVERSION_khz
    #df_comb['joules'] *= JOULE_CONVERSION

    return df1, df2, df_comb

def mcd_edp_plot(qps=200000, run_id=0, scale_requests=True, src_folder='aug19_mcdlogs', core_even=0, core_odd=0, get_stats=False, save_loc=None, silo=False):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

    x_offset, y_offset = 0.01/5, 0.01/5
    plt.figure(figsize=(9,7))

    if silo:
        df, dfr, outlier_list = start_mcdsilo_analysis('aug19_mcdsilologs/mcdsilo_combined.csv', scale_requests=True)
    else:
        df, dfr, outlier_list = start_mcd_analysis('aug10/mcd_combined.csv',
                                                    drop_outliers=True, 
                                                    scale_requests=scale_requests)

    #====================Linux Default==================
    sys = 'linux'
    itr = 1
    dvfs = '0xffff'
    rapl = 135

    df_runs = dfr[(dfr['sys']==sys) & (dfr['itr']==itr) & (dfr['dvfs']==dvfs) & (dfr['rapl']==rapl) & (dfr['QPS']==qps)]
    for idx, row in df_runs.iterrows():
        plt.plot([0, row['time']], [0, row['joules']], color=COLORS['linux_default'])

    df_even, df_odd = mcd_read_logfile_combined(sys, 
                                                run_id, 
                                                itr,
                                                dvfs,
                                                rapl,
                                                qps,
                                                nz=True,
                                                src_folder=src_folder,
                                                N=1,
                                                silo=silo)

    df1, df2, df_comb = mcd_prepare_for_edp(df_even, df_odd, core_even, core_odd)

    time_limit = dfr[(dfr["sys"]==sys) & (dfr["itr"]==itr) & (dfr["dvfs"]==dvfs) & (dfr["rapl"]==rapl) & (dfr["i"]==run_id) & (dfr["QPS"]==qps)]
    assert(time_limit.shape[0]==1)
    time_limit = time_limit.iloc[0]['time']

    df1 = df1[df1['timestamp'] <= time_limit]
    df2 = df2[df2['timestamp'] <= time_limit]
    df_comb = df_comb[df_comb['timestamp'] <= time_limit]

    dfl1, dfl2 = df1.copy(), df2.copy()
    J, T,_ = plot(df_comb, LABELS['linux_default'], projection=True, color=COLORS['linux_default'], include_edp_label=True, JOULE_CONVERSION=1, TIME_CONVERSION=1, plot_every=200)
    plt.text(x_offset + T, y_offset + J, f'(-, -, {rapl})')

    if get_stats:
        stats_ld = mcd_get_cumulative_statistics(sys, 
                                                 run_id, 
                                                 itr,
                                                 dvfs,
                                                 rapl,
                                                 qps,
                                                 src_folder=src_folder,
                                                 core_even=core_even,
                                                 core_odd=core_odd,
                                                 silo=silo)

    #bytes data
    df_even, df_odd = mcd_read_logfile_combined(sys, 
                                                run_id, 
                                                itr,
                                                dvfs,
                                                rapl,
                                                qps,
                                                nz=False,
                                                src_folder=src_folder,
                                                N=1,
                                                silo=silo)
    dfl1_orig, dfl2_orig, _ = mcd_prepare_for_edp(df_even, df_odd, core_even, core_odd)
    
    dfl1_orig = dfl1_orig[dfl1_orig["timestamp"] <= time_limit]
    dfl2_orig = dfl2_orig[dfl2_orig["timestamp"] <= time_limit]

    #====================Linux Tuned==================
    sys = 'linux'
    itr = 1
    dvfs = '0xffff'
    #rapl = 135
    
    d = df[(df["sys"]==sys) & (df["itr"]!=itr) & (df["dvfs"]!=dvfs) & (df["QPS"]==qps)]
    d = d[d['edp_mean']==d['edp_mean'].min()].iloc[0]

    sys = d['sys']
    itr = d['itr']
    dvfs = d['dvfs']
    rapl = d['rapl']

    df_runs = dfr[(dfr['sys']==sys) & (dfr['itr']==itr) & (dfr['dvfs']==dvfs) & (dfr['rapl']==rapl) & (dfr['QPS']==qps)]
    for idx, row in df_runs.iterrows():
        plt.plot([0, row['time']], [0, row['joules']], color=COLORS['linux_tuned'])

    df_even, df_odd = mcd_read_logfile_combined(sys, 
                                                run_id, 
                                                itr,
                                                dvfs,
                                                rapl,
                                                qps,
                                                nz=True,
                                                src_folder=src_folder,
                                                N=1,
                                                silo=silo)
    df1, df2, df_comb = mcd_prepare_for_edp(df_even, df_odd, core_even, core_odd)
 
    time_limit = dfr[(dfr["sys"]==sys) & (dfr["itr"]==itr) & (dfr["dvfs"]==dvfs) & (dfr["rapl"]==rapl) & (dfr["i"]==run_id) & (dfr["QPS"]==qps)]
    assert(time_limit.shape[0]==1)
    time_limit = time_limit.iloc[0]['time']

    df1 = df1[df1['timestamp'] <= time_limit]
    df2 = df2[df2['timestamp'] <= time_limit]
    df_comb = df_comb[df_comb['timestamp'] <= time_limit]

    dfl1_tuned, dfl2_tuned = df1.copy(), df2.copy()
    J, T,_ = plot(df_comb, LABELS['linux_tuned'], projection=True, color=COLORS['linux_tuned'], include_edp_label=True, JOULE_CONVERSION=1, TIME_CONVERSION=1, plot_every=200)
    plt.text(x_offset + T, y_offset + J, f'({d["itr"]}, {d["dvfs"]}, {d["rapl"]})')

    if get_stats:
        stats_lt = mcd_get_cumulative_statistics(sys, 
                                                 run_id, 
                                                 itr,
                                                 dvfs,
                                                 rapl,
                                                 qps,
                                                 src_folder=src_folder,
                                                 core_even=core_even,
                                                 core_odd=core_odd,
                                                 silo=silo)


    df_even, df_odd = mcd_read_logfile_combined(sys, 
                                                run_id, 
                                                itr,
                                                dvfs,
                                                rapl,
                                                qps,
                                                nz=False,
                                                src_folder=src_folder,
                                                N=1,
                                                silo=silo)
    dfl1_tuned_orig, dfl2_tuned_orig, _ = mcd_prepare_for_edp(df_even, df_odd, core_even, core_odd, simple=True)

    dfl1_tuned_orig = dfl1_tuned_orig[dfl1_tuned_orig["timestamp"] <= time_limit]
    dfl2_tuned_orig = dfl2_tuned_orig[dfl2_tuned_orig["timestamp"] <= time_limit]

    #if not get_stats:
    #    stats_ld, stats_lt = {}, {}

    #return dfl1, dfl2, dfl1_orig, dfl2_orig, dfl1_tuned, dfl2_tuned, dfl1_tuned_orig, dfl2_tuned_orig, stats_ld, stats_lt

    #====================EbbRT Tuned==================
    sys = 'ebbrt'
    itr = 1
    dvfs = '0xffff'
    #rapl = 135

    d = df[(df["sys"]==sys) & (df["itr"]!=itr) & (df["dvfs"]!=dvfs) & (df["QPS"]==qps)]
    d = d[d['edp_mean']==d['edp_mean'].min()].iloc[0]

    sys = d['sys']
    itr = d['itr']
    dvfs = d['dvfs']
    rapl = d['rapl']

    df_runs = dfr[(dfr['sys']==sys) & (dfr['itr']==itr) & (dfr['dvfs']==dvfs) & (dfr['rapl']==rapl) & (dfr['QPS']==qps)]
    for idx, row in df_runs.iterrows():
        plt.plot([0, row['time']], [0, row['joules']], color=COLORS['ebbrt_tuned'])

    df_even, df_odd = mcd_read_logfile_combined(sys, 
                                                run_id, 
                                                itr,
                                                dvfs,
                                                rapl,
                                                qps,
                                                nz=True,
                                                src_folder=src_folder,
                                                N=1,
                                                silo=silo)
    df1, df2, df_comb = mcd_prepare_for_edp(df_even, df_odd, core_even, core_odd)

    time_limit = dfr[(dfr["sys"]==sys) & (dfr["itr"]==itr) & (dfr["dvfs"]==dvfs) & (dfr["rapl"]==rapl) & (dfr["i"]==run_id) & (dfr["QPS"]==qps)]
    assert(time_limit.shape[0]==1)
    time_limit = time_limit.iloc[0]['time']

    df1 = df1[df1['timestamp'] <= time_limit]
    df2 = df2[df2['timestamp'] <= time_limit]
    df_comb = df_comb[df_comb['timestamp'] <= time_limit]

    dfe1, dfe2 = df1.copy(), df2.copy()
    J, T,_ = plot(df_comb, LABELS['ebbrt_tuned'], projection=True, color=COLORS['ebbrt_tuned'], include_edp_label=True, JOULE_CONVERSION=1, TIME_CONVERSION=1, plot_every=200)
    plt.text(x_offset + T, y_offset + J, f'({d["itr"]}, {d["dvfs"]}, {d["rapl"]})')

    if get_stats:
        stats_et = mcd_get_cumulative_statistics(sys, 
                                                 run_id, 
                                                 itr,
                                                 dvfs,
                                                 rapl,
                                                 qps,
                                                 src_folder=src_folder,
                                                 core_even=core_even,
                                                 core_odd=core_odd,
                                                 silo=silo)

    df_even, df_odd = mcd_read_logfile_combined(sys, 
                                                run_id, 
                                                itr,
                                                dvfs,
                                                rapl,
                                                qps,
                                                nz=False,
                                                src_folder=src_folder,
                                                N=1,
                                                silo=silo)

    dfe1_orig, dfe2_orig, _ = mcd_prepare_for_edp(df_even, df_odd, core_even, core_odd, simple=True)

    dfe1_orig = dfe1_orig[dfe1_orig["timestamp"] <= time_limit]
    dfe2_orig = dfe2_orig[dfe2_orig["timestamp"] <= time_limit]

    if silo:
        plt.title(f"Memcached Silo Workload \nwith 5,000,000 requests\nand QPS={qps}")
    else:
        plt.title(f"Memcached Workload \nwith 5,000,000 requests\nand QPS={qps}")
    prettify()

    if save_loc is not None:
        if silo:
            plt.savefig(f'{save_loc}/mcdsilo_edp_QPS{qps}.png')
        else:
            plt.savefig(f'{save_loc}/mcd_edp_QPS{qps}.png')

    if not get_stats:
        stats_ld, stats_lt, stats_et = {}, {}, {}

    return dfl1, dfl2, dfl1_orig, dfl2_orig, dfl1_tuned, dfl2_tuned, dfl1_tuned_orig, dfl2_tuned_orig, dfe1, dfe2, dfe1_orig, dfe2_orig, stats_ld, stats_lt, stats_et
    