import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import multiprocessing as mp
plt.ion()


JOULE_CONVERSION = 0.00001526 #counter * constant -> Joules
#TIME_CONVERSION = 0.00097656 # timestamp * constant -> sec
TIME_CONVERSION_NETPIPE = 1./ 10**6 #to get seconds
TIME_CONVERSION_NODEJS = 0.00097656 / 1000.
TIME_CONVERSION_khz = 1./(2899999*1000)

TPUT_CONVERSION_FACTOR = 2 * 8 / (1024*1024)

COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'llc_miss', 'joules', 'timestamp']
COLORS = {'linux_default': 'blue',
          'linux_tuned': 'green',
          'ebbrt_tuned': 'red'}          
LABELS = {'linux_default': 'Linux Default',
          'linux_tuned': 'Linux Tuned',
          'ebbrt_tuned': 'Library OS Tuned'}
COLORMAPS = {'linux_tuned': 'Greens',
             'ebbrt_tuned': 'Reds'}

#EbbRT MCD: /don-scratch/han/asplos_2021_datasets/mcd/ebbrt/ebbrt/ebbrt_mcd_refcycle_fix
#Linux MCD: /don-scratch/han/asplos_2021_datasets/mcd/linux/linux (max, min)

#CSV files generated as below:
#df = scan_all_data('jul20/governor/linux*dmesg*', column_names=COLS, N_THREADS=1)
#df = scan_all_data('jul20/rapl135/*dmesg*', column_names=COLS, N_THREADS=1)
def scan_all_data(name, column_names=None, TIME_CONVERSION=1, JOULE_CONVERSION=1, N_THREADS=1):
    '''Netpipe detailed logs -> summarized data
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

def mcdsilo_edp_calculator(df):
    conf = {}
    for qps in [50000, 100000, 200000]:
        for sys in ['linux', 'ebbrt']:
            d = df[(df['sys']==sys) & (df['QPS']==qps)].copy()

            d['edp'] = 0.5 * d['joules'] * d['time']

            conf[qps, sys] = d[d.edp==d.edp.min()].iloc[0]
    return conf

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

def write_mcd_table(df):
    '''TODO: some hard-coded values but ok
    '''

    df = df.copy()
    #df.tput_mean = df.tput_mean / (10**8)
    #df.tput_std = df.tput_std / (10**8)

    results = {'linux': {},
               'ebbrt': {}}
   
    eff_metric = 'joules'
    eff_metric_mean = f'{eff_metric}_mean'
    eff_metric_std = f'{eff_metric}_std'

    for sys in ['linux', 'ebbrt']:
        for msg in df['QPS'].unique():
            df2 = df[(df.sys==sys) & (df['QPS']==msg)].copy()

            if msg not in results[sys]:
                results[sys][msg] = {}

            if sys=='linux':
                df_pol_pol = df2[(df2.itr==1) & (df2.dvfs=='0xffff')]
                df_pol_nopol = df2[(df2.itr==1) & (df2.dvfs!='0xffff')]
                df_nopol_nopol = df2[(df2.itr!=1) & (df2.dvfs!='0xffff')]

                #val 1
                results['linux'][msg]['pol_pol_edp'] = (df_pol_pol.edp_mean.iloc[0], df_pol_pol.edp_std.iloc[0])
                results['linux'][msg][f'pol_pol_{eff_metric}'] = (df_pol_pol[eff_metric_mean].iloc[0], df_pol_pol[eff_metric_std].iloc[0])

                #val 2
                res = df_pol_nopol[df_pol_nopol.edp_mean==df_pol_nopol.edp_mean.min()]
                if res.shape[0] > 1: raise ValueError()
                res = res.iloc[0]

                results['linux'][msg]['pol_nopol_edp'] = (res.edp_mean, res.edp_std)
                results['linux'][msg][f'pol_nopol_{eff_metric}'] = (res[eff_metric_mean], res[eff_metric_std])

                #val 3: best tuned value for ITR and DVFS
                res = df_nopol_nopol[df_nopol_nopol.edp_mean==df_nopol_nopol.edp_mean.min()]
                if res.shape[0] > 1: raise ValueError()
                res = res.iloc[0]

                results['linux'][msg]['nopol_nopol_edp'] = (res.edp_mean, res.edp_std)
                results['linux'][msg][f'nopol_nopol_{eff_metric}'] = (res[eff_metric_mean], res[eff_metric_std])

                #results['linux'][msg]['nopol_nopol_best_edp'] = res.loc[['itr', 'dvfs', 'rapl']].to_dict()
                results['linux'][msg]['nopol_nopol_best_edp'] = df_nopol_nopol.sort_values(by='edp_mean', ascending=True).copy()

            elif sys=='ebbrt':
                df_nopol_nopol = df2[(df2.itr!=1) & (df2.dvfs!='0xffff')]

                #val 1: best value
                res = df_nopol_nopol[df_nopol_nopol.edp_mean==df_nopol_nopol.edp_mean.min()].iloc[0]
                results['ebbrt'][msg]['nopol_nopol_edp'] = (res.edp_mean, res.edp_std)
                results['ebbrt'][msg][f'nopol_nopol_{eff_metric}'] = (res[eff_metric_mean], res[eff_metric_std])

                #val 2: value for best linux parameters                                
                counter = 0
                for idx, row in results['linux'][msg]['nopol_nopol_best_edp'].iterrows():
                    res = df_nopol_nopol[(df_nopol_nopol['itr']==row['itr']) \
                                     & (df_nopol_nopol['dvfs']==row['dvfs']) \
                                     & (df_nopol_nopol['rapl']==row['rapl'])]

                    if res.shape[0] > 0:
                        print(f"Match found in row {counter}")
                        if counter!=0:
                            print(sys, msg, "Best params for linux not sampled in ebbrt")
                        res = res.iloc[0]

                        results['ebbrt'][msg]['linux_baseline_edp'] = (res.edp_mean, res.edp_std)
                        results['ebbrt'][msg][f'linux_baseline_{eff_metric}'] = (res[eff_metric_mean], res[eff_metric_std])
                        break
                    counter += 1

            else:
                raise ValueError()

    text_linux, text_ebbrt = [], []

    for msg in df['QPS'].unique():
        sep = '&'
        end = '\\\\ \hline'

        line_linux = f"Memcached: {msg} rps \
{sep} ${results['linux'][msg]['pol_pol_edp'][0]:.3f} \pm {results['linux'][msg]['pol_pol_edp'][1]:.3f}$ \
{sep} ${results['linux'][msg][f'pol_pol_{eff_metric}'][0]:.3f} \pm {results['linux'][msg][f'pol_pol_{eff_metric}'][1]:.3f}$ \
\
{sep} ${results['linux'][msg]['pol_nopol_edp'][0]:.3f} \pm {results['linux'][msg]['pol_nopol_edp'][1]:.3f}$ \
{sep} ${results['linux'][msg][f'pol_nopol_{eff_metric}'][0]:.3f} \pm {results['linux'][msg][f'pol_nopol_{eff_metric}'][1]:.3f}$ \
\
{sep} ${results['linux'][msg]['nopol_nopol_edp'][0]:.3f} \pm {results['linux'][msg]['nopol_nopol_edp'][1]:.3f}$ \
{sep} ${results['linux'][msg][f'nopol_nopol_{eff_metric}'][0]:.3f} \pm {results['linux'][msg][f'nopol_nopol_{eff_metric}'][1]:.3f}$ \
{end}"

        line_ebbrt = f"Memcached: {msg} rps \
{sep} ${results['ebbrt'][msg]['linux_baseline_edp'][0]:.3f} \pm {results['ebbrt'][msg]['linux_baseline_edp'][1]:.3f}$ \
{sep} ${results['ebbrt'][msg][f'linux_baseline_{eff_metric}'][0]:.3f} \pm {results['ebbrt'][msg][f'linux_baseline_{eff_metric}'][1]:.3f}$ \
\
{sep} ${results['ebbrt'][msg]['nopol_nopol_edp'][0]:.3f} \pm {results['ebbrt'][msg]['nopol_nopol_edp'][1]:.3f}$ \
{sep} ${results['ebbrt'][msg][f'nopol_nopol_{eff_metric}'][0]:.3f} \pm {results['ebbrt'][msg][f'nopol_nopol_{eff_metric}'][1]:.3f}$ \
{end}"
    
        text_linux.append(line_linux)
        text_ebbrt.append(line_ebbrt)

    return results, text_linux, text_ebbrt

def write_netpipe_table(df):
    '''TODO: some hard-coded values but ok
    '''

    df = df.copy()
    df.tput_mean = df.tput_mean / (10**7)
    df.tput_std = df.tput_std / (10**7)

    results = {'linux': {},
               'ebbrt': {}}

    eff_metric = 'tput'
    eff_metric_mean = f'{eff_metric}_mean'
    eff_metric_std = f'{eff_metric}_std'

    for sys in ['linux', 'ebbrt']:
        for msg in df.msg.unique():
            df2 = df[(df.sys==sys) & (df.msg==msg)].copy()

            if msg not in results[sys]:
                results[sys][msg] = {}


            if sys=='linux':
                df_pol_pol = df2[(df2.itr==1) & (df2.dvfs=='0xFFFF')]
                df_pol_nopol = df2[(df2.itr==1) & (df2.dvfs!='0xFFFF')]
                df_nopol_nopol = df2[(df2.itr!=1) & (df2.dvfs!='0xFFFF')]

                assert(df_pol_pol.shape[0]==1)
                assert(df_pol_nopol.shape[0]==14)
                assert(df_nopol_nopol.shape[0]==14*14)

                #val 1
                results['linux'][msg]['pol_pol_edp'] = (df_pol_pol.edp_mean.iloc[0], df_pol_pol.edp_std.iloc[0])
                results['linux'][msg][f'pol_pol_{eff_metric}'] = (df_pol_pol[eff_metric_mean].iloc[0], df_pol_pol[eff_metric_std].iloc[0])

                #val 2
                res = df_pol_nopol[df_pol_nopol.edp_mean==df_pol_nopol.edp_mean.min()]
                if res.shape[0] > 1: raise ValueError()
                res = res.iloc[0]

                results['linux'][msg]['pol_nopol_edp'] = (res.edp_mean, res.edp_std)
                results['linux'][msg][f'pol_nopol_{eff_metric}'] = (res[eff_metric_mean], res[eff_metric_std])

                #val 3: best tuned value for ITR and DVFS
                res = df_nopol_nopol[df_nopol_nopol.edp_mean==df_nopol_nopol.edp_mean.min()]
                if res.shape[0] > 1: raise ValueError()
                res = res.iloc[0]

                results['linux'][msg]['nopol_nopol_edp'] = (res.edp_mean, res.edp_std)
                results['linux'][msg][f'nopol_nopol_{eff_metric}'] = (res[eff_metric_mean], res[eff_metric_std])

                #results['linux'][msg]['nopol_nopol_best_edp'] = res.loc[['itr', 'dvfs', 'rapl']].to_dict()
                results['linux'][msg]['nopol_nopol_best_edp'] = df_nopol_nopol.sort_values(by='edp_mean', ascending=True).copy()

            elif sys=='ebbrt':
                df_nopol_nopol = df2[(df2.itr!=1) & (df2.dvfs!='0xFFFF')]
                assert(df_nopol_nopol.shape[0]==12*14)

                #val 1: best value
                res = df_nopol_nopol[df_nopol_nopol.edp_mean==df_nopol_nopol.edp_mean.min()].iloc[0]
                results['ebbrt'][msg]['nopol_nopol_edp'] = (res.edp_mean, res.edp_std)
                results['ebbrt'][msg][f'nopol_nopol_{eff_metric}'] = (res[eff_metric_mean], res[eff_metric_std])

                #val 2: value for best linux parameters                                
                counter = 0
                for idx, row in results['linux'][msg]['nopol_nopol_best_edp'].iterrows():
                    res = df_nopol_nopol[(df_nopol_nopol['itr']==row['itr']) \
                                     & (df_nopol_nopol['dvfs']==row['dvfs']) \
                                     & (df_nopol_nopol['rapl']==row['rapl'])]
                    if res.shape[0] > 0:
                        print(f"Match found in row {counter}")
                        if counter!=0:
                            print(sys, msg, "Best params for linux not sampled in ebbrt")
                        res = res.iloc[0]

                        results['ebbrt'][msg]['linux_baseline_edp'] = (res.edp_mean, res.edp_std)
                        results['ebbrt'][msg][f'linux_baseline_{eff_metric}'] = (res[eff_metric_mean], res[eff_metric_std])
                        break
                    counter += 1

                    '''
                res = df_nopol_nopol[(df_nopol_nopol['itr']==results['linux'][msg]['nopol_nopol_best_edp']['itr']) \
                                     & (df_nopol_nopol['dvfs']==results['linux'][msg]['nopol_nopol_best_edp']['dvfs']) \
                                     & (df_nopol_nopol['rapl']==results['linux'][msg]['nopol_nopol_best_edp']['rapl'])]
                
                if res.shape[0] == 1:
                    res = res.iloc[0]
                    results['ebbrt'][msg]['linux_baseline_edp'] = (res.edp_mean, res.edp_std)
                else: #best match params in linux not sampled for ebbrt

                    print(sys, msg, "Best params for linux not sampled in ebbrt")
                    
                    counter = 0
                    for idx, row in results['linux'][msg]['nopol_nopol_best_edp']:
                        res = df_nopol_nopol[(df_nopol_nopol['itr']==row['itr']) \
                                         & (df_nopol_nopol['dvfs']==row['dvfs']) \
                                         & (df_nopol_nopol['rapl']==row['rapl'])]
                        if res.shape[0] > 0:
                            print(f"Match found in row {counter}")
                            results['ebbrt'][msg]['linux_baseline_edp'] = (res.edp_mean, res.edp_std)
                            break
                        counter += 1
                    

                    #results['ebbrt'][msg]['linux_baseline_edp'] = (-1, -1)
                '''
            else:
                raise ValueError()

    text_linux, text_ebbrt = [], []

    for msg in df.msg.unique():
        sep = '&'
        end = '\\\\ \hline'

        line_linux = f"Netpipe: {msg} bytes \
{sep} ${results['linux'][msg]['pol_pol_edp'][0]:.3f} \pm {results['linux'][msg]['pol_pol_edp'][1]:.3f}$ \
{sep} ${results['linux'][msg][f'pol_pol_{eff_metric}'][0]:.3f} \pm {results['linux'][msg][f'pol_pol_{eff_metric}'][1]:.3f}$ \
\
{sep} ${results['linux'][msg]['pol_nopol_edp'][0]:.3f} \pm {results['linux'][msg]['pol_nopol_edp'][1]:.3f}$ \
{sep} ${results['linux'][msg][f'pol_nopol_{eff_metric}'][0]:.3f} \pm {results['linux'][msg][f'pol_nopol_{eff_metric}'][1]:.3f}$ \
\
{sep} ${results['linux'][msg]['nopol_nopol_edp'][0]:.3f} \pm {results['linux'][msg]['nopol_nopol_edp'][1]:.3f}$ \
{sep} ${results['linux'][msg][f'nopol_nopol_{eff_metric}'][0]:.3f} \pm {results['linux'][msg][f'nopol_nopol_{eff_metric}'][1]:.3f}$ \
{end}"

        line_ebbrt = f"Netpipe: {msg} bytes \
{sep} ${results['ebbrt'][msg]['linux_baseline_edp'][0]:.3f} \pm {results['ebbrt'][msg]['linux_baseline_edp'][1]:.3f}$ \
{sep} ${results['ebbrt'][msg][f'linux_baseline_{eff_metric}'][0]:.3f} \pm {results['ebbrt'][msg][f'linux_baseline_{eff_metric}'][1]:.3f}$ \
\
{sep} ${results['ebbrt'][msg]['nopol_nopol_edp'][0]:.3f} \pm {results['ebbrt'][msg]['nopol_nopol_edp'][1]:.3f}$ \
{sep} ${results['ebbrt'][msg][f'nopol_nopol_{eff_metric}'][0]:.3f} \pm {results['ebbrt'][msg][f'nopol_nopol_{eff_metric}'][1]:.3f}$ \
{end}"
    
        text_linux.append(line_linux)
        text_ebbrt.append(line_ebbrt)

    return results, text_linux, text_ebbrt

def write_nodejs_table(df):
    df = df.copy()

    df.tput_mean = df.tput_mean / (10**3)
    df.tput_std = df.tput_std / (10**3)

    results = {'linux': {},
               'ebbrt': {}}

    eff_metric = 'tput'
    eff_metric_mean = f'{eff_metric}_mean'
    eff_metric_std = f'{eff_metric}_std'

    for sys in ['linux', 'ebbrt']:
        df2 = df[(df.sys==sys)].copy()

        if sys=='linux':
            df_pol_pol = df2[(df2.itr==1) & (df2.dvfs=='0xffff')]
            df_pol_nopol = df2[(df2.itr==1) & (df2.dvfs!='0xffff')]
            df_nopol_nopol = df2[(df2.itr!=1) & (df2.dvfs!='0xffff')]


            #val 1
            results['linux']['pol_pol_edp'] = (df_pol_pol.edp_mean.iloc[0], df_pol_pol.edp_std.iloc[0])
            results['linux'][f'pol_pol_{eff_metric}'] = (df_pol_pol[eff_metric_mean].iloc[0], df_pol_pol[eff_metric_std].iloc[0])

            #val 2
            #res = df_pol_nopol[df_pol_nopol.edp_mean==df_pol_nopol.edp_mean.min()]
            res = df_pol_nopol[df_pol_nopol.eput_mean==df_pol_nopol.eput_mean.max()]
            if res.shape[0] > 1: raise ValueError()
            res = res.iloc[0]

            results['linux']['pol_nopol_edp'] = (res.edp_mean, res.edp_std)
            results['linux'][f'pol_nopol_{eff_metric}'] = (res[eff_metric_mean], res[eff_metric_std])

            #val 3: best tuned value for ITR and DVFS
            #res = df_nopol_nopol[df_nopol_nopol.edp_mean==df_nopol_nopol.edp_mean.min()]
            res = df_nopol_nopol[df_nopol_nopol.eput_mean==df_nopol_nopol.eput_mean.max()]
            if res.shape[0] > 1: raise ValueError()
            res = res.iloc[0]

            results['linux']['nopol_nopol_edp'] = (res.edp_mean, res.edp_std)
            results['linux'][f'nopol_nopol_{eff_metric}'] = (res[eff_metric_mean], res[eff_metric_std])

            #results['linux'][msg]['nopol_nopol_best_edp'] = res.loc[['itr', 'dvfs', 'rapl']].to_dict()
            results['linux']['nopol_nopol_best_edp'] = df_nopol_nopol.sort_values(by='edp_mean', ascending=True).copy()
            results['linux'][f'nopol_nopol_best_{eff_metric}'] = df_nopol_nopol.sort_values(by=f'{eff_metric}_mean', ascending=False).copy()


        elif sys=='ebbrt':
            df_nopol_nopol = df2[(df2.itr!=1) & (df2.dvfs!='0xFFFF')]
            #assert(df_nopol_nopol.shape[0]==12*14)

            #val 1: best value
            res = df_nopol_nopol[df_nopol_nopol.eput_mean==df_nopol_nopol.eput_mean.max()].iloc[0]
            results['ebbrt']['nopol_nopol_edp'] = (res.edp_mean, res.edp_std)
            results['ebbrt'][f'nopol_nopol_{eff_metric}'] = (res[eff_metric_mean], res[eff_metric_std])

            #val 2: value for best linux parameters                                
            counter = 0
            for idx, row in results['linux'][f'nopol_nopol_best_{eff_metric}'].iterrows():
                res = df_nopol_nopol[(df_nopol_nopol['itr']==row['itr']) \
                                 & (df_nopol_nopol['dvfs']==row['dvfs']) \
                                 & (df_nopol_nopol['rapl']==row['rapl'])]
                if res.shape[0] > 0:
                    print(f"Match found in row {counter}")
                    if counter!=0:
                        print(sys, msg, "Best params for linux not sampled in ebbrt")
                    res = res.iloc[0]

                    results['ebbrt']['linux_baseline_edp'] = (res.edp_mean, res.edp_std)
                    results['ebbrt'][f'linux_baseline_{eff_metric}'] = (res[eff_metric_mean], res[eff_metric_std])
                    break
                counter += 1

        else:
            raise ValueError()

    text_linux, text_ebbrt = [], []

    sep = '&'
    end = '\\\\ \hline'

    line_linux = f"NodeJS: \
{sep} ${results['linux']['pol_pol_edp'][0]:.3f} \pm {results['linux']['pol_pol_edp'][1]:.3f}$ \
{sep} ${results['linux'][f'pol_pol_{eff_metric}'][0]:.3f} \pm {results['linux'][f'pol_pol_{eff_metric}'][1]:.3f}$ \
\
{sep} ${results['linux']['pol_nopol_edp'][0]:.3f} \pm {results['linux']['pol_nopol_edp'][1]:.3f}$ \
{sep} ${results['linux'][f'pol_nopol_{eff_metric}'][0]:.3f} \pm {results['linux'][f'pol_nopol_{eff_metric}'][1]:.3f}$ \
\
{sep} ${results['linux']['nopol_nopol_edp'][0]:.3f} \pm {results['linux']['nopol_nopol_edp'][1]:.3f}$ \
{sep} ${results['linux'][f'nopol_nopol_{eff_metric}'][0]:.3f} \pm {results['linux'][f'nopol_nopol_{eff_metric}'][1]:.3f}$ \
{end}"

    line_ebbrt = f"NodeJS: \
{sep} ${results['ebbrt']['linux_baseline_edp'][0]:.3f} \pm {results['ebbrt']['linux_baseline_edp'][1]:.3f}$ \
{sep} ${results['ebbrt'][f'linux_baseline_{eff_metric}'][0]:.3f} \pm {results['ebbrt'][f'linux_baseline_{eff_metric}'][1]:.3f}$ \
\
{sep} ${results['ebbrt']['nopol_nopol_edp'][0]:.3f} \pm {results['ebbrt']['nopol_nopol_edp'][1]:.3f}$ \
{sep} ${results['ebbrt'][f'nopol_nopol_{eff_metric}'][0]:.3f} \pm {results['ebbrt'][f'nopol_nopol_{eff_metric}'][1]:.3f}$ \
{end}"
    
    text_linux.append(line_linux)
    text_ebbrt.append(line_ebbrt)

    return results, text_linux, text_ebbrt

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

#512k -> linux -> 7762 (default)
#512k -> linux -> 8385.032203 (tuned)

'''
df_even  = [read_mcd_logfile_combined(f'aug19_mcdlogs/mcd_dmesg.0_{2*c}_1_0xffff_135_200000', nz=True) for c in range(8)]
df_odd  = [read_mcd_logfile_combined(f'aug19_mcdlogs/mcd_dmesg.0_{2*c+1}_1_0xffff_135_200000', nz=True) for c in range(8)]
a = pd.concat(df_even, axis=0).sort_values(by='timestamp', ascending=True)
b = pd.concat(df_odd, axis=0).sort_values(by='timestamp', ascending=True)
'''
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


    #return dfl_1, dfl_2, dfl_tuned_1, dfl_tuned_2, dfe_1, dfe_2


def mcd_bar_plot(stats_ld, stats_lt, stats_et, save_loc, qps, silo=False):

    metric_labels = ['Instructions', 'Energy', 'Ref. Cycles', 'Transmitted Bytes', 'Transmitted Bytes/Joule', 'Interrupts']
    N_metrics = len(metric_labels) #number of clusters
    N_systems = 3 #number of plot loops

    fig = plt.figure()
    ax = fig.subplots()

    idx = np.arange(N_metrics) #one group per metric
    width = 0.2

    df_dict = {'linux_default': stats_ld,
               'linux_tuned': stats_lt,
               'ebbrt_tuned': stats_et}

    data_dict = {}

    for sys in df_dict: #compute metrics
        data_dict[sys] = np.array([df_dict[sys]['instructions'],
                                   df_dict[sys]['joules'],
                                   df_dict[sys]['cycles'],
                                   df_dict[sys]['txbytes'],
                                   df_dict[sys]['txbytes'] / df_dict[sys]['joules'],
                                   df_dict[sys]['interrupts']
                                  ])

    counter = 0
    for sys in data_dict: #normalize and plot
        data = data_dict[sys] / data_dict['linux_default']

        ax.bar(idx + counter*width, data, width, label=LABELS[sys], color=COLORS[sys])
        counter += 1

    ax.set_xticks(idx)
    ax.set_xticklabels(metric_labels, rotation=15, fontsize='small')
    ax.set_ylabel('Metric / Metric for Linux Default')
    plt.legend()

    if save_loc is not None:
        if silo:
            plt.savefig(f'{save_loc}/mcdsilo_combined_barplot_QPS{qps}.png')
        else:
            plt.savefig(f'{save_loc}/mcd_combined_barplot_QPS{qps}.png')

def mcd_nonidle_plot_df(df):
    df = df.copy()

    for c in ['instructions', 'refcyc', 'cycles', 'llc_miss', 'joules', 'timestamp']: 
        df[c] = df[c] - df[c].min()

    df2 = df[['instructions', 'refcyc', 'cycles', 'joules', 'timestamp', 'i']].diff()
    df2.columns = [f'{c}_diff' for c in df2.columns]

    df = pd.concat([df, df2], axis=1)

    print("Assuming >= 1 second time")
    for c in ['instructions', 'refcyc', 'cycles', 'llc_miss', 'joules', 'timestamp']: 
        df[c] = df[c] - df[c].min()

    #df['time'] = df['timestamp'] * TIME_CONVERSION_khz
    #df = df[df['time'] < time_limit].copy()

    #df['timestamp_diff'] *= TIME_CONVERSION_khz
    df['nonidle_timestamp_diff'] = df['refcyc_diff'] * TIME_CONVERSION_khz
    df['nonidle_frac_diff'] = df['nonidle_timestamp_diff'] / df['timestamp_diff']

    return df

def mcd_nonidle_plot(qps, dfl, dfl_tuned, dfe, save_loc, silo=False):
    #dfl1, dfl2, dfl1_orig, dfl2_orig, dfl1_tuned, dfl2_tuned, dfl1_tuned_orig, dfl2_tuned_orig, dfe1, dfe2, dfe1_orig, dfe2_orig, stats_ld, stats_lt, stats_et = mcd_edp_plot(qps=qps, run_id=0, get_stats=False)

    plt.figure()

    x = mcd_nonidle_plot_df(dfl)
    plt.plot(x['timestamp'], x['nonidle_frac_diff'],'p', c=COLORS['linux_default'], label=LABELS['linux_default'])

    x = mcd_nonidle_plot_df(dfl_tuned)
    plt.plot(x['timestamp'], x['nonidle_frac_diff'],'p', c=COLORS['linux_tuned'], label=LABELS['linux_tuned'])

    x = mcd_nonidle_plot_df(dfe)
    plt.plot(x['timestamp'], x['nonidle_frac_diff'],'p', c=COLORS['ebbrt_tuned'], label=LABELS['ebbrt_tuned'])

    plt.grid()

    plt.xlabel('Time (s)')
    plt.ylabel('Non-idle Time (%)')
    plt.ylim((0, 1.001))
    plt.title(f"Non-idle Fractional Time for Memcached\nwith QPS={qps}")
    plt.legend()

    if silo:
        plt.savefig(f'{save_loc}/mcdsilo_nonidle_QPS{qps}.png')
    else:
        plt.savefig(f'{save_loc}/mcd_nonidle_QPS{qps}.png')

'''
def mcd_nonidle_plot(filename, time_limit=0):
    df = read_mcd_logfile(filename, nz=True)

    for c in ['instructions', 'refcyc', 'cycles', 'llc_miss', 'joules', 'timestamp']: 
        df[c] = df[c] - df[c].min()

    df2 = df[['instructions', 'refcyc', 'cycles', 'joules', 'timestamp', 'i']].diff()
    df2.columns = [f'{c}_diff' for c in df2.columns]

    df = pd.concat([df, df2], axis=1)

    print("Assuming >= 1 second time")
    for c in ['instructions', 'refcyc', 'cycles', 'llc_miss', 'joules', 'timestamp']: 
        df[c] = df[c] - df[c].min()

    df['time'] = df['timestamp'] * TIME_CONVERSION_khz
    #df = df[df['time'] < time_limit].copy()

    df['timestamp_diff'] *= TIME_CONVERSION_khz
    df['nonidle_timestamp_diff'] = df['refcyc_diff'] * TIME_CONVERSION_khz
    df['nonidle_frac_diff'] = df['nonidle_timestamp_diff'] / df['timestamp_diff']

    return df
'''

def mcd_get_latency_joule_curve(save_loc=None, qps=600000, silo=False):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

    if silo:
        df, dfr, outlier_list = start_mcdsilo_analysis('aug19_mcdsilologs/mcdsilo_combined.csv', scale_requests=True)
    else:
        df, dfr, outlier_list = start_mcd_analysis('aug10/mcd_combined.csv', drop_outliers=True, scale_requests=True)

    plt.clf()
    plt.figure()

    #linux default
    if not silo:
        x = dfr[(dfr.sys=='linux') & (dfr.QPS==qps) & (dfr.itr==1) & (dfr.dvfs=='0xffff') & (dfr.rapl==135)][['sys', 'itr', 'rapl', 'dvfs', 'QPS', 'joules', 'time', 'read_10th', 'read_50th', 'read_90th', 'read_99th']]
        x['tradeoff'] = (x['read_99th'] * x['joules'])
        x.sort_values(by='joules', ascending=True, inplace=True)
        x = x.iloc[0]
        plt.plot(x['read_99th'], x['joules'], 'p', c=COLORS['linux_default'], label=LABELS['linux_default'])
        plt.plot([x['read_99th'], 0], [x['joules'], x['joules']], '--', c=COLORS['linux_default'])
        plt.plot([x['read_99th'], x['read_99th']], [x['joules'], 0], '--', c=COLORS['linux_default'])

    #linux tuned
    x = dfr[(dfr.sys=='linux') & (dfr.QPS==qps) & (dfr.itr!=1) & (dfr.dvfs!='0xffff')][['sys', 'itr', 'rapl', 'dvfs', 'QPS', 'joules', 'time', 'read_10th', 'read_50th', 'read_90th', 'read_99th']]
    x['tradeoff'] = (x['read_99th'] * x['joules'])
    x.sort_values(by='tradeoff', ascending=True, inplace=True)
    y = x[(x['read_99th']<270) | (x['joules']<750)]

    plt.plot(y['read_99th'], y['joules'], 'p', label=LABELS['linux_tuned'], c=COLORS['linux_tuned'])

    #ebbrt tuned
    x = dfr[(dfr.sys=='ebbrt') & (dfr.QPS==qps) & (dfr.itr!=1) & (dfr.dvfs!='0xffff')][['sys', 'itr', 'rapl', 'dvfs', 'QPS', 'joules', 'time', 'read_10th', 'read_50th', 'read_90th', 'read_99th']]
    x['tradeoff'] = (x['read_99th'] * x['joules'])
    x.sort_values(by='tradeoff', ascending=True, inplace=True)
    #y = x[(x['read_99th']<270) | (x['joules']<750)]
    y = x

    plt.plot(y['read_99th'], y['joules'], 'p', label=LABELS['ebbrt_tuned'], c=COLORS['ebbrt_tuned'])

    plt.xlim(0, 550)
    plt.ylim(0, 1200)
    plt.xlabel('99th Tail Latency ($\mu$s)')
    plt.ylabel('Energy (Joules)')
    plt.title(f'Energy-Latency Tuning Curves with QPS={qps}')
    plt.grid()
    plt.legend()

    if save_loc is not None:
        plt.savefig(f'{save_loc}/mcd_energy_latency_tuning_curves_{qps}.png')

def mcd_exploratory_plots(dfl, dfl_tuned, dfe, slice_middle=False, save_loc=None):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

    #dfl, dfl_orig, dfl_tuned, dfl_tuned_orig, dfe, dfe_orig = asplos_nodejs_edp_plots(save_loc=None, slice_middle=slice_middle)

    #for m in ['rx_bytes', 'rx_desc', 'tx_bytes', 'tx_desc']:
    '''
    for m in ['txbytes', 'rxbytes']:
        plt.figure()
        plt.plot(dfl_orig['timestamp'], dfl_orig[m], 'p', c=COLORS['linux_default'], label=LABELS['linux_default'])
        plt.plot(dfl_tuned_orig['timestamp'], dfl_tuned_orig[m], 'p', c=COLORS['linux_tuned'], label=LABELS['linux_tuned'])
        plt.plot(dfe_orig['timestamp'], dfe_orig[m], 'p', c=COLORS['ebbrt_tuned'], label=LABELS['ebbrt_tuned'])
        plt.xlabel('Time (s)')
        
        plt.legend()
        plt.grid()

        if m=='txbytes':
            counts = dfl_tuned_orig['txbytes'].value_counts()
            x_loc = dfl_tuned['timestamp'].max() * 1.1

            for packet_size in counts.index:
                packet_count = counts.loc[packet_size]

                #plt.text(x_loc, packet_size, f'(size={packet_size}, counts={packet_count})', c=COLORS['linux_tuned'])

            plt.ylabel('Transmitted Bytes')
            plt.title("Timeline plot for Transmitted Bytes")

        elif m=='rxbytes':
            plt.ylabel('Received Bytes')
            plt.title("Timeline plot for Received Bytes")

        if save_loc:
            plt.savefig(f'{save_loc}/nodejs_timeline_{m}.png')

    dfl['ins_per_ref_cycle'] = dfl['instructions_diff'] / dfl['refcyc_diff']
    dfl_tuned['ins_per_ref_cycle'] = dfl_tuned['instructions_diff'] / dfl_tuned['refcyc_diff']
    dfe['ins_per_ref_cycle'] = dfe['instructions_diff'] / dfe['refcyc_diff']
    '''

    #for m in ['joules_diff', 'llc_miss_diff', 'instructions_diff', 'ref_cycles_diff', 'ins_per_ref_cycle']:
    for m in ['joules_diff']:
        plt.figure()
        plt.plot(dfl['timestamp'], dfl[m], 'p', c=COLORS['linux_default'], label=LABELS['linux_default'])
        plt.plot(dfl_tuned['timestamp'], dfl_tuned[m], 'p', c=COLORS['linux_tuned'], label=LABELS['linux_tuned'])
        plt.plot(dfe['timestamp'], dfe[m], 'p', c=COLORS['ebbrt_tuned'], label=LABELS['ebbrt_tuned'])
        
        m = m.replace('_diff', '')

        if m=='ins_per_ref_cycle':
            continue

        #d = dfl_orig[dfl_orig[m]==0]
        #plt.plot(d['timestamp'], d[m], 'p', c=COLORS['linux_default'], marker='+')
        
        #d = dfl_tuned_orig[dfl_tuned_orig[m]==0]
        #plt.plot(d['timestamp'], d[m], 'p', c=COLORS['linux_tuned'], marker='+')
        
        #d = dfe_orig[dfe_orig[m]==0]
        #plt.plot(d['timestamp'], d[m], 'p', c=COLORS['ebbrt_tuned'], marker='+')

        plt.xlabel('Time (s)')
        
        plt.legend()
        plt.grid()
        
        plt.arrow(0.8, 0.018, 0.1, 0.005, width=0.0003)
        x = dfl[(dfl.joules_diff>0.0) & (dfl.joules_diff<0.02)]
        idle_power = x.joules_diff.sum() / (x.shape[0] * x['timestamp_diff'].mean())
        #plt.text(0.9 + 0.001, 0.023 + 0.001, f'Idle Power: {idle_power:.2f} Watts')

        if m=='joules':
            plt.ylabel('Energy Consumed (Joules)')
            plt.title('Timeline plot for Energy Consumed')
        
        if save_loc:
            plt.savefig(f'{save_loc}/mcd_timeline_{m}.png')


def mcd_get_cumulative_statistics(sys, 
                                  run_id, 
                                  itr,
                                  dvfs,
                                  rapl,
                                  qps,
                                  src_folder='aug19_mcdlogs',
                                  core_even=0,
                                  core_odd=0,
                                  silo=False):


    if not silo:
        _, dfr, _ = start_mcd_analysis('aug10/mcd_combined.csv',
                                       drop_outliers=True, 
                                       scale_requests=True)
    else:
        _, dfr, _ = start_mcdsilo_analysis('aug19_mcdsilologs/mcdsilo_combined.csv', scale_requests=True)

    time_limit = dfr[(dfr["sys"]==sys) & (dfr["itr"]==itr) & (dfr["dvfs"]==dvfs) & (dfr["rapl"]==rapl) & (dfr["i"]==run_id) & (dfr["QPS"]==qps)]
    assert(time_limit.shape[0]==1)
    time_limit = time_limit.iloc[0]['time']

    df_even, df_odd = mcd_read_logfile_combined(sys, run_id, itr, dvfs, rapl, qps, nz=False, src_folder=src_folder, N=8, silo=silo)

    stats = {'instructions': 0,
             'cycles': 0,
             'txbytes': 0,
             'interrupts': 0}

    for d in df_even + df_odd:
        d['timestamp'] -= d['timestamp'].min()
        d['timestamp'] *= TIME_CONVERSION_khz
        d = d[d['timestamp'] <= time_limit]

        stats['instructions'] += (d['instructions'].max() - d['instructions'].min())
        stats['cycles'] += (d['refcyc'].max() - d['refcyc'].min())
        stats['txbytes'] += d['txbytes'].sum() #(d['txbytes'].max() - d['txbytes'].min())
        stats['interrupts'] += (d.shape[0])

    df_even, df_odd = mcd_read_logfile_combined(sys, run_id, itr, dvfs, rapl, qps, nz=True, src_folder=src_folder, N=8, silo=silo)
    df1, df2, df_comb = mcd_prepare_for_edp(df_even, df_odd, core_even, core_odd)
    df1 = df1[df1['timestamp'] <= time_limit]    
    df2 = df2[df2['timestamp'] <= time_limit]    
    df_comb = df_comb[df_comb['timestamp'] <= time_limit]    

    stats['joules'] = df_comb.tail(1).iloc[0]['joules']

    return stats

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

def mcd_get_tsc_filename(filename):
    fname_split = filename.split('.')

    if filename.find('ebbrt')>-1:
        assert(len(fname_split)==3)
    
    else:
        assert(len(fname_split)==2)

    fname_split[0] = fname_split[0].replace('_dmesg', '_rdtsc')
    tag_split = fname_split[1].split('_')

    tag = '_'.join([tag_split[0]] + tag_split[2:])

    tsc_filename = f'{fname_split[0]}.{tag}'

    return tsc_filename

def mcd_get_time_bounds(filename):
    if filename.find('mcdsilo_')>-1:
        with open(filename) as f:
            lines = f.readlines()

        start = np.max([int(l.rstrip('\n').split()[2]) for l in lines])
        return start, -1
        #for l in lines:
        #    _, _, start, end = [int(x) for x in l.rstrip('\n').split()]

            #if end <= start:
            #    continue

            #delta = (end-start) * TIME_CONVERSION_khz
            #print(start, end, delta)
            #if abs(delta - 20.0) < 2:
            #    return start, end

    if filename.find('ebbrt')>-1:
        with open(filename) as f:
            lines = f.readlines()
        assert(len(lines)==1)
        lines = lines[0]

        start, end = [int(x) for x in lines.rstrip('\n').split()]
        print('Time Limits for EbbRT:', start, end)

        return start, end

    else:
        df = pd.read_csv(filename, sep=' ', names=[0,1,2,3])


    df[2] = df[2].astype(int)
    df[3] = df[3].astype(int)

    start =  df[2].max()
    end = df[3].min()

    if start < end:
        return start, end
    else:
        df['diff'] = (df[3] - df[2]) * TIME_CONVERSION_khz
        df = df[(df['diff']-30).abs() < 5].copy()
    
        start =  df[2].max()
        end = df[3].min()
        assert(start < end)
        return start, end

def make_3d_plots_netpipe(scale_requests, workload, save_loc):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

    df, dfr, outlier_list = start_netpipe_analysis(drop_outliers=True)

    min_edp_run = dfr.groupby(['sys', 'itr', 'dvfs', 'rapl', 'msg'])['edp'].min().reset_index()
    
    best_runs = pd.merge(dfr, min_edp_run, on=['sys', 'itr', 'dvfs', 'rapl', 'msg', 'edp'], how='inner')

    xmax = 0

    for msg in [8192]:
        for sys in ['linux_tuned', 'ebbrt_tuned', 'linux_default']:
        #for msg in [64, 8192, 65536, 524288]:
            if sys=='linux_default':
                d = best_runs[(best_runs['sys']=='linux') & (best_runs['msg']==msg) & (best_runs['itr']==1) & (best_runs['dvfs']=='0xFFFF')]
                assert(d.shape[0]==1)

            elif sys=='linux_tuned':
                d = best_runs[(best_runs['sys']=='linux') & (best_runs['msg']==msg) & (best_runs['itr']!=1) & (best_runs['dvfs']!='0xFFFF')]

            elif sys=='ebbrt_tuned':
                d = best_runs[(best_runs['sys']=='ebbrt') & (best_runs['msg']==msg) & (best_runs['itr']!=1) & (best_runs['dvfs']!='0xFFFF')]

            d = d.sort_values(by='edp', ascending=True)

            xmax = max(xmax, d.shape[0])

            if sys=='linux_default':
                plt.hlines(d.iloc[0]['edp'], 0, xmax, colors=COLORS[sys], linestyle='--', label=LABELS[sys])

            else:
                plt.plot(d['edp'].tolist(), 'p', c=COLORS[sys], label=LABELS[sys])

            plt.grid()
            plt.legend()
            plt.xlabel('Configuration ranked by EDP')
            plt.ylabel('EDP (Js)')
            plt.title(f'Configurations ranked by EDP for Netpipe\nfor 5000 Msg Size={msg}\nDataset Size = 3753 Experiments')

            if save_loc:
                plt.savefig(f'{save_loc}/plots3d_ranked_netpipe_msg{msg}.png')

    #crude but works
    fig = plt.figure(figsize=(9,7))
    for msg in [8192]:
        for sys in ['linux_tuned', 'ebbrt_tuned', 'linux_default']:
        #for msg in [64, 8192, 65536, 524288]:
            if sys=='linux_default':
                d = best_runs[(best_runs['sys']=='linux') & (best_runs['msg']==msg) & (best_runs['itr']==1) & (best_runs['dvfs']=='0xFFFF')]
                assert(d.shape[0]==1)

            elif sys=='linux_tuned':
                d = best_runs[(best_runs['sys']=='linux') & (best_runs['msg']==msg) & (best_runs['itr']!=1) & (best_runs['dvfs']!='0xFFFF')]

            elif sys=='ebbrt_tuned':
                d = best_runs[(best_runs['sys']=='ebbrt') & (best_runs['msg']==msg) & (best_runs['itr']!=1) & (best_runs['dvfs']!='0xFFFF')]

            d = d.sort_values(by='edp', ascending=True)

            plt.plot(d['time'], d['joules'], 'p', c=COLORS[sys], label=LABELS[sys])
            plt.xlabel('Time (sec)')
            plt.ylabel('Energy (Joules)')
            plt.legend()
            plt.title(f"Number of Experiments Time-Energy Space")
            plt.grid()

    if save_loc is not None:
        plt.savefig(f'{save_loc}/plots3d_netpipe_scatter_points.png')

    #crude but works
    for msg in [8192]:
        for sys in ['linux_tuned', 'ebbrt_tuned', 'linux_default']:
        #for msg in [64, 8192, 65536, 524288]:
            if sys=='linux_default':
                d = best_runs[(best_runs['sys']=='linux') & (best_runs['msg']==msg) & (best_runs['itr']==1) & (best_runs['dvfs']=='0xFFFF')]
                assert(d.shape[0]==1)

            elif sys=='linux_tuned':
                d = best_runs[(best_runs['sys']=='linux') & (best_runs['msg']==msg) & (best_runs['itr']!=1) & (best_runs['dvfs']!='0xFFFF')]

            elif sys=='ebbrt_tuned':
                d = best_runs[(best_runs['sys']=='ebbrt') & (best_runs['msg']==msg) & (best_runs['itr']!=1) & (best_runs['dvfs']!='0xFFFF')]

            d = d.sort_values(by='edp', ascending=True)



            if sys=='linux_default': 
                continue
            '''
            fig = plt.figure(figsize=(9,7))
            ax = fig.add_subplot(111, projection='3d')    

            hist, xedges, yedges = np.histogram2d(d['time'], d['joules'], bins=[20,10])

            xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
            xpos = xpos.ravel()
            ypos = ypos.ravel()
            zpos = 0

            dx = dy = 0.5 * np.ones_like(zpos)
            dz = hist.ravel()

            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=COLORS[sys])

            plt.xlabel('Time (sec)')
            plt.ylabel('Energy (Joules)')

            if sys=='linux_tuned':
                plt.title(f"Number of Experiments in Time-Energy Space for Linux")
            elif sys=='ebbrt_tuned':
                plt.title(f"Number of Experiments in Time-Energy Space for Library OS")
            '''

            fig = plt.figure(figsize=(9,7))
            ax = fig.add_subplot(111)        

            hist, xedges, yedges = np.histogram2d(d['time'], d['joules'], bins=[20,10])
            ax.imshow(hist[::-1, :], cmap=COLORMAPS[sys])
            plt.xlabel('Time (sec)')
            plt.ylabel('Energy (Joules)')

            if sys=='linux_tuned':
                plt.title(f"Number of Experiments Time-Energy Space for Linux")            

            elif sys=='ebbrt_tuned':
                plt.title(f"Number of Experiments Time-Energy Space for Library OS")


            ax.set_xticklabels([f'{x:.2f}' for x in xedges])
            ax.set_yticklabels([f'{y:.2f}' for y in yedges[::-1]], rotation=90)

            for t in range(hist.shape[1]):
                for e in range(hist.shape[0]):
                    if int(hist[e,t])>0:
                        ax.text(t, hist.shape[0] - e - 1, int(hist[e, t]), ha="center", va="center", color="w")

            if save_loc is not None:
                plt.savefig(f'{save_loc}/plots3d_netpipe_heatmap_{sys}.png')


def make_3d_plots_nodejs(scale_requests, workload, save_loc):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
    
    df, dfr, outlier_list = start_nodejs_analysis('aug4/nodejs_8_4.csv', 
                                                   drop_outliers=True, 
                                                   scale_requests=scale_requests)


    min_edp_run = dfr.groupby(['sys', 'itr', 'dvfs', 'rapl'])['edp'].min().reset_index()
    
    best_runs = pd.merge(dfr, min_edp_run, on=['sys', 'itr', 'dvfs', 'rapl', 'edp'], how='inner')

    xmax = 0
    plt.figure(figsize=(9,8))

    for sys in ['linux_tuned', 'ebbrt_tuned', 'linux_default']:
        if sys=='linux_default':
            d = best_runs[(best_runs['sys']=='linux') & (best_runs['msg']==msg) & (best_runs['itr']==1) & (best_runs['dvfs']=='0xFFFF')]
            assert(d.shape[0]==1)

        elif sys=='linux_tuned':
            d = best_runs[(best_runs['sys']=='linux') & (best_runs['msg']==msg) & (best_runs['itr']!=1) & (best_runs['dvfs']!='0xFFFF')]

        elif sys=='ebbrt_tuned':
            d = best_runs[(best_runs['sys']=='ebbrt') & (best_runs['msg']==msg) & (best_runs['itr']!=1) & (best_runs['dvfs']!='0xFFFF')]

        d = d.sort_values(by='edp', ascending=True)

        xmax = max(xmax, d.shape[0])


        if sys=='linux_default':
            plt.hlines(d.iloc[0]['edp'], 0, xmax, colors=COLORS[sys], linestyle='--', label=LABELS[sys])

        else:
            plt.plot(d['edp'].tolist(), 'p', c=COLORS[sys], label=LABELS[sys])

        plt.grid()
        plt.legend()
        plt.xlabel('Configuration ranked by EDP')
        plt.ylabel('EDP (Js)')
        plt.title(f'Configurations ranked by EDP for Netpipe\nfor 5000 Msg Size={msg}\nDataset Size = 3753 Experiments')

        if save_loc:
            plt.savefig(f'{save_loc}/plots3d_ranked_netpipe_msg{msg}.png')

    #crude but works
    for msg in [8192]:
        for sys in ['linux_tuned', 'ebbrt_tuned', 'linux_default']:
        #for msg in [64, 8192, 65536, 524288]:
            if sys=='linux_default':
                d = best_runs[(best_runs['sys']=='linux') & (best_runs['msg']==msg) & (best_runs['itr']==1) & (best_runs['dvfs']=='0xFFFF')]
                assert(d.shape[0]==1)

            elif sys=='linux_tuned':
                d = best_runs[(best_runs['sys']=='linux') & (best_runs['msg']==msg) & (best_runs['itr']!=1) & (best_runs['dvfs']!='0xFFFF')]

            elif sys=='ebbrt_tuned':
                d = best_runs[(best_runs['sys']=='ebbrt') & (best_runs['msg']==msg) & (best_runs['itr']!=1) & (best_runs['dvfs']!='0xFFFF')]

            d = d.sort_values(by='edp', ascending=True)



            if sys=='linux_default': 
                continue
            fig = plt.figure(figsize=(9,7))
            ax = fig.add_subplot(111, projection='3d')    

            hist, xedges, yedges = np.histogram2d(d['time'], d['joules'], bins=[20,10])

            xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
            xpos = xpos.ravel()
            ypos = ypos.ravel()
            zpos = 0

            dx = dy = 0.5 * np.ones_like(zpos)
            dz = hist.ravel()

            ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', color=COLORS[sys])

            plt.xlabel('Time (sec)')
            plt.ylabel('Energy (Joules)')

            plt.title(f"Number of Experiments in Time-Energy Space for {sys.replace('_', ' ')}")

            fig = plt.figure(figsize=(9,7))
            plt.plot(d['time'], d['joules'], 'p', c=COLORS[sys], label=LABELS[sys])
            plt.xlabel('Time (sec)')
            plt.ylabel('Energy (Joules)')

            plt.title(f"Number of Experiments Time-Energy Space for {sys.replace('_', ' ')}")

            fig = plt.figure(figsize=(9,7))
            ax = fig.add_subplot(111)        

            hist, xedges, yedges = np.histogram2d(d['time'], d['joules'], bins=[20,10])
            ax.imshow(hist[::-1, :], cmap=COLORMAPS[sys])
            plt.xlabel('Time (sec)')
            plt.ylabel('Energy (Joules)')
            plt.title(f"Number of Experiments Time-Energy Space for {sys.replace('_', ' ')}")            

            ax.set_xticklabels([f'{x:.2f}' for x in xedges])
            ax.set_yticklabels([f'{y:.2f}' for y in yedges[::-1]], rotation=90)

            for t in range(hist.shape[1]):
                for e in range(hist.shape[0]):
                    ax.text(t, hist.shape[0] - e - 1, int(hist[e, t]), ha="center", va="center", color="w")


    #df, dfr, outlier_list = start_mcd_analysis('aug10/mcd_combined.csv',
    #                                            drop_outliers=True, 
    #                                            scale_requests=scale_requests)




def asplos_mcd_edp_plots(save_loc, slice_middle=False, scale_requests=True, qps=200000):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

    #src_folder = 'aug13_nodejslogs'
    src_folder = 'aug19_mcdlogs'
    run_id = 1
    x_offset, y_offset = 0.01/5, 0.01/5

    plt.figure(figsize=(9,7))

    #this is temporary without log data
    df, dfr, outlier_list = start_mcd_analysis('aug18/linux_8_9.csv', drop_outliers=True, scale_requests=scale_requests)

    #1a: linux default
    d = df[(df['sys']=='linux') & (df['itr']==1) & (df['dvfs']=='0xffff') & (df['rapl']==135) & (df['qps']==qps)]
    assert(d.shape[0]==1)
    d = d.iloc[0]

    filename = f'{src_folder}/mcd_dmesg.{run_id}_*_{d["itr"]}_{d["dvfs"]}_{d["rapl"]}_{d["qps"]}' #* for 16 cores
    print(filename)
    df_log = read_mcd_logfile_combined(filename)
    time_limit = dfr[(dfr["sys"]==d["sys"]) & (dfr["itr"]==d["itr"]) & (dfr["dvfs"]==d["dvfs"]) & (dfr["rapl"]==d["rapl"]) & (dfr["i"]==run_id) & (dfr["qps"]==qps)]
    assert(time_limit.shape[0]) 
    time_limit = time_limit.iloc[0]["time"]
    df_log = df_log[df_log['timestamp'] <= time_limit]

    #dfl, dfl_orig = process_nodejs_logs(df_log, slice_middle=slice_middle)
    dfl = df_log.copy()

    J, T,_ = plot(df_log, LABELS['linux_default'], projection=True, color=COLORS['linux_default'], include_edp_label=True, JOULE_CONVERSION=1, TIME_CONVERSION=1, plot_every=200)
    plt.text(x_offset + T, y_offset + J, f'(-, -, {d["rapl"]})')

    #1b: linux tuned
    d = df[(df['sys']=='linux') & (df['itr']!=1) & (df['dvfs']!='0xffff') & (df['qps']==qps)]
    d = d[d['edp_mean']==d['edp_mean'].min()].iloc[0]

    filename = f'{src_folder}/mcd_dmesg.{run_id}_*_{d["itr"]}_{d["dvfs"]}_{d["rapl"]}_{d["qps"]}' #* for 16 cores
    print(filename)
    df_log = read_mcd_logfile_combined(filename)
    time_limit = dfr[(dfr["sys"]==d["sys"]) & (dfr["itr"]==d["itr"]) & (dfr["dvfs"]==d["dvfs"]) & (dfr["rapl"]==d["rapl"]) & (dfr["i"]==run_id) & (dfr["qps"]==qps)]
    assert(time_limit.shape[0]) 
    time_limit = time_limit.iloc[0]["time"]
    df_log = df_log[df_log['timestamp'] <= time_limit]

    #dfl_tuned, dfl_tuned_orig = process_nodejs_logs(df_log, slice_middle=slice_middle)
    dfl_tuned = df_log.copy()

    J, T,_ = plot(df_log, LABELS['linux_tuned'], projection=True, color=COLORS['linux_tuned'], include_edp_label=True, JOULE_CONVERSION=1, TIME_CONVERSION=1, plot_every=200)
    plt.text(x_offset + T, y_offset + J, f'({d["itr"]}, {d["dvfs"]}, {d["rapl"]})')

    #1c: ebbrt 
    d = df[(df['sys']=='ebbrt') & (df['itr']!=1) & (df['dvfs']!='0xffff') & (df['qps']==qps)]
    d = d[d['edp_mean']==d['edp_mean'].min()].iloc[0]

    filename = f'{src_folder}/ebbrt_dmesg.{run_id}_*_{d["itr"]}_{d["dvfs"]}_{d["rapl"]}_{d["qps"]}' #* for 16 cores
    print(filename)
    df_log = read_mcd_logfile_combined(filename)
    time_limit = dfr[(dfr["sys"]==d["sys"]) & (dfr["itr"]==d["itr"]) & (dfr["dvfs"]==d["dvfs"]) & (dfr["rapl"]==d["rapl"]) & (dfr["i"]==run_id) & (dfr["qps"]==qps)]
    assert(time_limit.shape[0]) 
    time_limit = time_limit.iloc[0]["time"]
    df_log = df_log[df_log['timestamp'] <= time_limit]

    #dfl_tuned, dfl_tuned_orig = process_nodejs_logs(df_log, slice_middle=slice_middle)
    dfe = df_log.copy()

    J, T,_ = plot(df_log, LABELS['ebbrt_tuned'], projection=True, color=COLORS['ebbrt_tuned'], include_edp_label=True, JOULE_CONVERSION=1, TIME_CONVERSION=1, plot_every=200)
    plt.text(x_offset + T, y_offset + J, f'({d["itr"]}, {d["dvfs"]}, {d["rapl"]})')

    prettify()

    plt.title("Memcached Workload \nwith 5,000,000 requests")

    if save_loc is not None:
        plt.savefig(f'{save_loc}/mcd_edp_detailed_qps{qps}.png')

    #return dfl, dfl_orig, dfl_tuned, dfl_tuned_orig, dfe, dfe_orig
    return dfl, dfl_tuned, dfe


def asplos_mcdsilo_edp_plots_aggregated(save_loc=None, scale_requests=True, qps=50000, drop_outliers=False):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

    x_offset, y_offset = 0.01/5, 0.01/5

    plt.figure(figsize=(9,7))

    #this is temporary without log data
    #df, dfr, outlier_list = start_nodejs_analysis('aug4/nodejs_8_4.csv', drop_outliers=True, scale_requests=True)
    df, dfr, outlier_list = start_mcdsilo_analysis('aug19_mcdsilologs/mcdsilo_combined.csv', scale_requests=True, drop_outliers=drop_outliers) 

    #1a: linux default
    d = df[(df['sys']=='linux') & (df['QPS']==qps)] 
    T = d["time_mean"].iloc[0]
    J = d["joules_mean"].iloc[0]
    edp_val = 0.5*T*J
    plt.plot([0, T], [0, J], 'p-', label=f'{LABELS["linux_default"]} EDP={edp_val:.2f}', c=COLORS['linux_default'])
    plt.text(x_offset + T, y_offset + J, f'(-, -, {d["rapl"].iloc[0]})')

    #1b: linux tuned
    #d = df[(df['sys']=='linux') & (df['itr']!=1) & (df['dvfs']!='0xffff')] 
    d = df[(df['sys']=='linux') & (df['QPS']==qps)]
    d = d[d['edp_mean']==d['edp_mean'].min()]
    T = d["time_mean"].iloc[0]
    J = d["joules_mean"].iloc[0]
    edp_val = 0.5*T*J
    plt.plot([0, T], [0, J], 'p-', label=f'{LABELS["linux_tuned"]} EDP={edp_val:.2f}', c=COLORS['linux_tuned'])
    plt.text(x_offset + T, y_offset + J, f'({d["itr"].iloc[0]}, {d["dvfs"].iloc[0]}, {d["rapl"].iloc[0]})')


    #1c: ebbrt 
    #d = df[(df['sys']=='ebbrt') & (df['itr']!=1) & (df['dvfs']!='0xffff')]
    d = df[(df['sys']=='ebbrt') & (df['QPS']==qps)]
    d = d[d['edp_mean']==d['edp_mean'].min()]
    T = d["time_mean"].iloc[0]
    J = d["joules_mean"].iloc[0]
    edp_val = 0.5*T*J
    plt.plot([0, T], [0, J], 'p-', label=f'{LABELS["ebbrt_tuned"]} EDP={edp_val:.2f}', c=COLORS['ebbrt_tuned'])
    plt.text(x_offset + T, y_offset + J, f'({d["itr"].iloc[0]}, {d["dvfs"].iloc[0]}, {d["rapl"].iloc[0]})')

    prettify()

    plt.title(f"Memcached Silo Workload \nwith QPS={qps}")

    if save_loc is not None:
        plt.savefig(f'{save_loc}/mcdsilo_edp_aggregated_qps{qps}.png')


def asplos_mcd_edp_plots_aggregated(save_loc=None, scale_requests=True, qps=200000):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

    x_offset, y_offset = 0.01/5, 0.01/5

    plt.figure(figsize=(9,7))

    #this is temporary without log data
    #df, dfr, outlier_list = start_mcd_analysis('aug18/linux_8_9.csv', drop_outliers=True, scale_requests=scale_requests)
    df, dfr, outlier_list = start_mcd_analysis('aug10/mcd_combined.csv', drop_outliers=True, scale_requests=scale_requests)

    #1a: linux default
    d = df[(df['sys']=='linux') & (df['itr']==1) & (df['dvfs']=='0xffff') & (df['rapl']==135) & (df['QPS']==qps)] 
    T = d["time_mean"].iloc[0]
    J = d["joules_mean"].iloc[0]
    edp_val = 0.5*T*J
    plt.plot([0, T], [0, J], 'p-', label=f'{LABELS["linux_default"]} EDP={edp_val:.2f}', c=COLORS['linux_default'])
    plt.text(x_offset + T, y_offset + J, f'(-, -, {d["rapl"].iloc[0]})')

    #1b: linux tuned
    d = df[(df['sys']=='linux') & (df['itr']!=1) & (df['dvfs']!='0xffff') & (df['QPS']==qps)] 
    d = d[d['edp_mean']==d['edp_mean'].min()]
    T = d["time_mean"].iloc[0]
    J = d["joules_mean"].iloc[0]
    edp_val = 0.5*T*J
    plt.plot([0, T], [0, J], 'p-', label=f'{LABELS["linux_tuned"]} EDP={edp_val:.2f}', c=COLORS['linux_tuned'])
    plt.text(x_offset + T, y_offset + J, f'({d["itr"].iloc[0]}, {d["dvfs"].iloc[0]}, {d["rapl"].iloc[0]})')


    #1c: ebbrt 
    d = df[(df['sys']=='ebbrt') & (df['itr']!=1) & (df['dvfs']!='0xffff') & (df['QPS']==qps)]
    d = d[d['edp_mean']==d['edp_mean'].min()]
    T = d["time_mean"].iloc[0]
    J = d["joules_mean"].iloc[0]
    edp_val = 0.5*T*J
    plt.plot([0, T], [0, J], 'p-', label=f'{LABELS["ebbrt_tuned"]} EDP={edp_val:.2f}', c=COLORS['ebbrt_tuned'])
    plt.text(x_offset + T, y_offset + J, f'({d["itr"].iloc[0]}, {d["dvfs"].iloc[0]}, {d["rapl"].iloc[0]})')

    prettify()

    plt.title(f"Memcached Workload \nwith 5,000,000 requests\nand QPS={qps}")

    if save_loc is not None:
        plt.savefig(f'{save_loc}/mcd_edp_aggregated_qps{qps}.png')


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


def asplos_nodejs_edp_plots(save_loc, slice_middle=False, scale_requests=True):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

    #src_folder = 'aug13_nodejslogs'
    #src_folder = 'aug14_nodejslogs'
    src_folder = 'aug19_nodejslogs' #tx bytes fixed
    run_id = 1
    x_offset, y_offset = 0.01/5, 0.01/5

    plt.figure(figsize=(9,7))

    #this is temporary without log data
    df, dfr, outlier_list = start_nodejs_analysis('aug4/nodejs_8_4.csv', drop_outliers=True, scale_requests=scale_requests)

    #1a: linux default
    d = df[(df['sys']=='linux') & (df['itr']==1) & (df['dvfs']=='0xffff')].iloc[0]
    filename = f'{src_folder}/node_dmesg.{run_id}_1_{d["itr"]}_{d["dvfs"]}_{d["rapl"]}'
    df_log = read_nodej_logfile(filename)
    time_limit = dfr[(dfr["sys"]==d["sys"]) & (dfr["itr"]==d["itr"]) & (dfr["dvfs"]==d["dvfs"]) & (dfr["rapl"]==d["rapl"]) & (dfr["i"]==run_id)]
    assert(time_limit.shape[0]) 
    time_limit = time_limit.iloc[0]["time"]
    df_log = df_log[df_log['timestamp'] <= time_limit]
    
    #dfl = df_log.copy()
    dfl, dfl_orig = process_nodejs_logs(df_log, slice_middle=slice_middle)

    J, T,_ = plot(df_log, LABELS['linux_default'], projection=True, color=COLORS['linux_default'], include_edp_label=True, JOULE_CONVERSION=1, TIME_CONVERSION=1, plot_every=200)
    #edp_val = 0.5*T*J
    #plt.plot([0, T], [0, J], 'p-', label=f'Linux Default EDP={edp_val:.2f}', c=COLORS['linux_default'])
    plt.text(x_offset + T, y_offset + J, f'(-, -, {d["rapl"]})')

    #1b: linux tuned
    d = df[(df['sys']=='linux') & (df['itr']!=1) & (df['dvfs']!='0xffff')]
    d = d[d['edp_mean']==d['edp_mean'].min()].iloc[0]
    filename = f'{src_folder}/node_dmesg.{run_id}_1_{d["itr"]}_{d["dvfs"]}_{d["rapl"]}'
    df_log = read_nodej_logfile(filename)

    time_limit = dfr[(dfr["sys"]==d["sys"]) & (dfr["itr"]==d["itr"]) & (dfr["dvfs"]==d["dvfs"]) & (dfr["rapl"]==d["rapl"]) & (dfr["i"]==run_id)]
    assert(time_limit.shape[0])
    time_limit = time_limit.iloc[0]["time"]
    df_log = df_log[df_log['timestamp'] <= time_limit]
    #dfl_tuned = df_log.copy()
    dfl_tuned, dfl_tuned_orig = process_nodejs_logs(df_log, slice_middle=slice_middle)

    J, T,_ = plot(df_log, LABELS['linux_tuned'], projection=True, color=COLORS['linux_tuned'], include_edp_label=True, JOULE_CONVERSION=1, TIME_CONVERSION=1, plot_every=200)
    #edp_val = 0.5*T*J
    #plt.plot([0, T], [0, J], 'p-', label=f'Linux tuned EDP={edp_val:.2f}', c=COLORS['linux_tuned'])
    plt.text(x_offset + T, y_offset + J, f'({d["itr"]}, {d["dvfs"]}, {d["rapl"]})')


    #1c: ebbrt 
    d = df[(df['sys']=='ebbrt') & (df['itr']!=1) & (df['dvfs']!='0xffff')]
    d = d[d['edp_mean']==d['edp_mean'].min()].iloc[0]

    filename = f'{src_folder}/ebbrt_dmesg.{run_id}_1_{d["itr"]}_{d["dvfs"]}_{d["rapl"]}.csv'
    df_log = read_nodej_logfile(filename)

    time_limit = dfr[(dfr["sys"]==d["sys"]) & (dfr["itr"]==d["itr"]) & (dfr["dvfs"]==d["dvfs"]) & (dfr["rapl"]==d["rapl"]) & (dfr["i"]==run_id)]
    assert(time_limit.shape[0])
    time_limit = time_limit.iloc[0]["time"]
    df_log = df_log[df_log['timestamp'] <= time_limit]
    dfe = df_log.copy()
    dfe, dfe_orig = process_nodejs_logs(df_log, slice_middle=slice_middle)

    J, T,_ = plot(df_log, LABELS['ebbrt_tuned'], projection=True, color=COLORS['ebbrt_tuned'], include_edp_label=True, JOULE_CONVERSION=1, TIME_CONVERSION=1, plot_every=200)
    #edp_val = 0.5*T*J
    #plt.plot([0, T], [0, J], 'p-', label=f'EbbRT tuned EDP={edp_val:.2f}', c=COLORS['ebbrt_tuned'])
    plt.text(x_offset + T, y_offset + J, f'({d["itr"]}, {d["dvfs"]}, {d["rapl"]})')

    prettify()

    plt.title("NodeJS Workload \nwith 100,000 requests")

    if save_loc is not None:
        plt.savefig(f'{save_loc}/nodejs_edp_detailed.png')

    return dfl, dfl_orig, dfl_tuned, dfl_tuned_orig, dfe, dfe_orig

def asplos_nodejs_edp_plots_aggregated(save_loc):
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    x_offset, y_offset = 0.01/5, 0.01/5

    plt.figure(figsize=(9,7))

    #this is temporary without log data
    df, dfr, outlier_list = start_nodejs_analysis('aug4/nodejs_8_4.csv', drop_outliers=True, scale_requests=True)

    #1a: linux default
    d = df[(df['sys']=='linux') & (df['itr']==1) & (df['dvfs']=='0xffff')] 
    T = d["time_mean"].iloc[0]
    J = d["joules_mean"].iloc[0]
    edp_val = 0.5*T*J
    plt.plot([0, T], [0, J], 'p-', label=f'{LABELS["linux_default"]} EDP={edp_val:.2f}', c=COLORS['linux_default'])
    plt.text(x_offset + T, y_offset + J, f'(-, -, {d["rapl"].iloc[0]})')

    #1b: linux tuned
    d = df[(df['sys']=='linux') & (df['itr']!=1) & (df['dvfs']!='0xffff')] 
    d = d[d['edp_mean']==d['edp_mean'].min()]
    T = d["time_mean"].iloc[0]
    J = d["joules_mean"].iloc[0]
    edp_val = 0.5*T*J
    plt.plot([0, T], [0, J], 'p-', label=f'{LABELS["linux_tuned"]} EDP={edp_val:.2f}', c=COLORS['linux_tuned'])
    plt.text(x_offset + T, y_offset + J, f'({d["itr"].iloc[0]}, {d["dvfs"].iloc[0]}, {d["rapl"].iloc[0]})')


    #1c: ebbrt 
    d = df[(df['sys']=='ebbrt') & (df['itr']!=1) & (df['dvfs']!='0xffff')]
    d = d[d['edp_mean']==d['edp_mean'].min()]
    T = d["time_mean"].iloc[0]
    J = d["joules_mean"].iloc[0]
    edp_val = 0.5*T*J
    plt.plot([0, T], [0, J], 'p-', label=f'{LABELS["ebbrt_tuned"]} EDP={edp_val:.2f}', c=COLORS['ebbrt_tuned'])
    plt.text(x_offset + T, y_offset + J, f'({d["itr"].iloc[0]}, {d["dvfs"].iloc[0]}, {d["rapl"].iloc[0]})')

    prettify()

    plt.title("NodeJS Workload with 100,000 requests")

    plt.savefig(f'{save_loc}/nodejs_edp.png')

def asplos_nodejs_latency_plots(save_loc):
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    plt.figure(figsize=(9,7))

    #this is temporary without log data
    df, dfr, outlier_list = start_nodejs_analysis('aug4/nodejs_8_4.csv', drop_outliers=True, scale_requests=True)

    #1a: linux default
    d = df[(df['sys']=='linux') & (df['itr']==1) & (df['dvfs']=='0xffff')] 
    edp_val = (0.5*d["time_mean"]*d["joules_mean"]).iloc[0]
    plt.plot()

    #1b: linux tuned
    d = df[(df['sys']=='linux') & (df['itr']!=1) & (df['dvfs']!='0xffff')] 
    d = d[d['edp_mean']==d['edp_mean'].min()]
    edp_val = (0.5*d["time_mean"]*d["joules_mean"]).iloc[0]
    plt.plot([0, d['time_mean']], [0, d['joules_mean']], 'p-', label=f'{LABELS["linux_tuned"]} EDP={edp_val:.2f}', c=COLORS['linux_tuned'])

    #1c: ebbrt 
    d = df[(df['sys']=='ebbrt') & (df['itr']!=1) & (df['dvfs']!='0xffff')]
    d = d[d['edp_mean']==d['edp_mean'].min()]
    edp_val = (0.5*d["time_mean"]*d["joules_mean"]).iloc[0]
    plt.plot([0, d['time_mean']], [0, d['joules_mean']], 'p-', label=f'{LABELS["ebbrt_tuned"]} EDP={edp_val:.2f}', c=COLORS['ebbrt_tuned'])

    prettify()

    plt.title("NodeJS Workload with 100,000 requests")

    plt.savefig(f'{save_loc}/nodejs_edp.png')

def asplos_nodejs_log_analysis(save_loc=None, slice_middle=False):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

    run_id = 1

    #prepare data
    dfl, dfl_orig, dfl_tuned, dfl_tuned_orig, dfe, dfe_orig = asplos_nodejs_edp_plots(save_loc=None, slice_middle=slice_middle)

    #busy plot
    plt.figure(figsize=(9,7))
    plt.plot(dfl['timestamp'], dfl['nonidle_frac_diff'], 'p-', label=LABELS['linux_default'])
    plt.plot(dfl_tuned['timestamp'], dfl_tuned['nonidle_frac_diff'], 'p-', label=LABELS['linux_tuned'], c=COLORS['linux_tuned'])
    plt.plot(dfe['timestamp'], dfe['nonidle_frac_diff'], 'p-', label=LABELS['ebbrt_tuned'], c=COLORS['ebbrt_tuned'])
    
    plt.xlabel('Time (s)')
    plt.ylabel('Non-idle Time (%)')
    plt.ylim((0, 1.001))
    if slice_middle:
        plt.title(f"Non-idle Fractional Time in 0.2sec window")
    else:
        plt.title(f"Non-idle Fractional Time")

    plt.grid()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), 
               loc='lower left', 
               #mode="expand", 
               borderaxespad=0.)
    
    if save_loc is not None:
        if slice_middle: 
            plt.savefig(f'{save_loc}/nodejs_nonidle_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/nodejs_nonidle.png')

    #instructions
    plt.figure()
    plt.bar([1,2,3], [dfl['instructions'].tail(1).iloc[0], \
                      dfl_tuned['instructions'].tail(1).iloc[0], \
                      dfe['instructions'].tail(1).iloc[0]],
                      color = [COLORS['linux_default'], COLORS['linux_tuned'], COLORS['ebbrt_tuned']])
    plt.xticks(ticks=[1,2,3], labels = [LABELS['linux_default'], LABELS['linux_tuned'], LABELS['ebbrt_tuned']])
    plt.ylabel('Total Instructions')
    if slice_middle:
        plt.title(f"Total Instructions in 0.2sec window from 1 sample run")
    else:
        plt.title(f"Total Instructions from 1 sample run")

    if save_loc is not None:
        if slice_middle:
            plt.savefig(f'{save_loc}/nodejs_instructions_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/nodejs_instructions_.png')

    #joules
    plt.figure()
    plt.bar([1,2,3], [dfl['joules'].tail(1).iloc[0], \
                      dfl_tuned['joules'].tail(1).iloc[0], \
                      dfe['joules'].tail(1).iloc[0]],
                      color = [COLORS['linux_default'], COLORS['linux_tuned'], COLORS['ebbrt_tuned']])
    plt.xticks(ticks=[1,2,3], labels = [LABELS['linux_default'], LABELS['linux_tuned'], LABELS['ebbrt_tuned']])
    plt.ylabel('Energy Used (Joules)')
    if slice_middle:
        plt.title(f"Energy Consumed in 0.2sec window from 1 sample run")
    else:
        plt.title(f"Total Energy Consumed from 1 sample run")
    
    if save_loc is not None:
        if slice_middle:
            plt.savefig(f'{save_loc}/nodejs_energy_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/nodejs_energy.png')

    #bytes transmitted
    plt.figure()
    plt.bar([1,2,3], [dfl_orig['txbytes'].sum(), \
                      dfl_tuned_orig['txbytes'].sum(), \
                      dfe_orig['txbytes'].sum()],
                      color = [COLORS['linux_default'], COLORS['linux_tuned'], COLORS['ebbrt_tuned']])
    plt.xticks(ticks=[1,2,3], labels = [LABELS['linux_default'], LABELS['linux_tuned'], LABELS['ebbrt_tuned']])
    plt.ylabel('Data Transmitted (Bytes)')
    if slice_middle:
        plt.title(f"Data Transmitted in 0.2sec window from 1 sample run")
    else:
        plt.title(f"Total Data Transmitted from 1 sample run")

    if save_loc is not None:
        if slice_middle:
            plt.savefig(f'{save_loc}/nodejs_txbytes_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/nodejs_txbytes_.png')

    #transmission efficiency
    plt.figure()
    plt.bar([1,2,3], [dfl_orig['txbytes'].sum() / dfl['joules'].tail(1).iloc[0], \
                      dfl_tuned_orig['txbytes'].sum() / dfl_tuned['joules'].tail(1).iloc[0], \
                      dfe_orig['txbytes'].sum() / dfe['joules'].tail(1).iloc[0]],
                      color = [COLORS['linux_default'], COLORS['linux_tuned'], COLORS['ebbrt_tuned']])
    plt.xticks(ticks=[1,2,3], labels = [LABELS['linux_default'], LABELS['linux_tuned'], LABELS['ebbrt_tuned']])
    plt.ylabel('Data Transmitted / Energy (Bytes / Joule)')
    if slice_middle:
        plt.title(f"Data Transmission Efficiency in 0.2sec window from 1 sample run")
    else:
        plt.title(f"Data Transmission Efficiency from 1 sample run")

    if save_loc is not None:
        if slice_middle:
            plt.savefig(f'{save_loc}/nodejs_instructionsperjoule__window200ms.png')
        else:
            plt.savefig(f'{save_loc}/nodejs_instructionsperjoule_.png')

    #cluster all bars together
    metric_labels = ['Instructions', 'Energy', 'Ref. Cycles', 'Transmitted Bytes', 'Received Bytes', 'Transmitted Bytes/Joule', 'Interrupts']
    N_metrics = len(metric_labels) #number of clusters
    N_systems = 3 #number of plot loops

    fig = plt.figure()
    ax = fig.subplots()

    idx = np.arange(N_metrics) #one group per metric
    width = 0.2

    df_dict = {'linux_default': (dfl, dfl_orig),
               'linux_tuned': (dfl_tuned, dfl_tuned_orig),
               'ebbrt_tuned': (dfe, dfe_orig)}
    data_dict = {}

    for sys in df_dict: #compute metrics
        data_dict[sys] = np.array([df_dict[sys][0]['instructions'].tail(1).iloc[0],
                                   df_dict[sys][0]['joules'].tail(1).iloc[0],
                                   df_dict[sys][0]['refcyc'].tail(1).iloc[0],
                                   df_dict[sys][1]['txbytes'].sum(),
                                   df_dict[sys][1]['rxbytes'].sum(),
                                   df_dict[sys][1]['txbytes'].sum() / df_dict[sys][0]['joules'].tail(1).iloc[0],
                                   #df_dict[sys][0]['joules'].tail(1).iloc[0] / df_dict[sys][1].shape[0],
                                   df_dict[sys][1].shape[0]
                                  ])

    counter = 0
    for sys in data_dict: #normalize and plot
        data = data_dict[sys] / data_dict['linux_default']

        ax.bar(idx + counter*width, data, width, label=LABELS[sys], color=COLORS[sys])
        counter += 1

    ax.set_xticks(idx)
    ax.set_xticklabels(metric_labels, rotation=15, fontsize='small')
    ax.set_ylabel('Metric / Metric for Linux Default')
    plt.legend()

    if slice_middle:
        plt.title(f"Relative Metrics in 0.2sec window from 1 sample run")
    else:
        plt.title(f"Relative Metrics from 1 sample run")

    if save_loc is not None:
        if slice_middle:
            plt.savefig(f'{save_loc}/nodejs_combined_barplot_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/nodejs_combined_barplot.png')



    #stacked energy plot

    '''
    3 systems
    total energy -> dfl, dfl_tuned, dfe
    busy time -> see non-idle-frac
    energy calc: idle_perc * total_time * base_power
    stacking: levels -> 1. total/realized, 2. idle power = 0
    '''

    #stacked interrupt delays

    N_systems = 3 #number of plot loops

    fig = plt.figure()
    ax = fig.subplots()

    idx = np.arange(N_systems) #one group per metric
    width = 0.2

    df_dict = {'linux_default': (dfl, dfl_orig),
               'linux_tuned': (dfl_tuned, dfl_tuned_orig),
               'ebbrt_tuned': (dfe, dfe_orig)}
    data_dict = {}

    keys = list(df_dict.keys())

    power_idle = {'linux_default': 17,
                  'linux_tuned': 17,
                  'ebbrt_tuned': 17}

    joules = np.array([df_dict[sys][0]['joules'].tail(1).iloc[0] for sys in keys])
    
    joules_idle = []
    for sys in keys:
        total_time = df_dict[sys][0]['timestamp'].tail(1).iloc[0]
        
        idle_perc = 1 - df_dict[sys][0]['nonidle_frac_diff'].mean()

        d = total_time * idle_perc * power_idle[sys]

        joules_idle.append(d)

    joules_idle = np.array(joules_idle)

    ax.bar(idx, joules_idle, width, label='Estimated Idling Energy Consumption')
    plt.plot(idx, joules, 'p', marker='+', label='Measured Total Energy Consumption')
    ax.bar(idx, joules-joules_idle, width, bottom=joules_idle, label='Measured Minus Idle Estimate', alpha=0.2)

    ax.set_xticks(idx)
    labels = [LABELS[l] for l in keys]
    ax.set_xticklabels(labels, rotation=0, fontsize='small')
    ax.set_ylabel("Joules")
    plt.title('Energy Consumption Breakdown')
    plt.legend()

    if slice_middle:
        plt.title(f"Energy breakdown by time in 0.2sec window from 1 sample run")
    else:
        plt.title(f"Energy breakdown by time from 1 sample run")

    if save_loc is not None:
        if slice_middle:
            plt.savefig(f'{save_loc}/nodejs_energy_breakdown_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/nodejs_energy_breakdown.png')


    return dfl, dfl_tuned, dfe, dfl_orig, dfl_tuned_orig, dfe_orig

def asplos_nodejs_exploratory_plots(slice_middle=False, save_loc=None):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

    dfl, dfl_orig, dfl_tuned, dfl_tuned_orig, dfe, dfe_orig = asplos_nodejs_edp_plots(save_loc=None, slice_middle=slice_middle)

    #for m in ['rx_bytes', 'rx_desc', 'tx_bytes', 'tx_desc']:
    for m in ['txbytes', 'rxbytes']:
        plt.figure()
        plt.plot(dfl_orig['timestamp'], dfl_orig[m], 'p', c=COLORS['linux_default'], label=LABELS['linux_default'])
        plt.plot(dfl_tuned_orig['timestamp'], dfl_tuned_orig[m], 'p', c=COLORS['linux_tuned'], label=LABELS['linux_tuned'])
        plt.plot(dfe_orig['timestamp'], dfe_orig[m], 'p', c=COLORS['ebbrt_tuned'], label=LABELS['ebbrt_tuned'])
        plt.xlabel('Time (s)')
        
        plt.legend()
        plt.grid()

        if m=='txbytes':
            counts = dfl_tuned_orig['txbytes'].value_counts()
            x_loc = dfl_tuned['timestamp'].max() * 1.1

            for packet_size in counts.index:
                packet_count = counts.loc[packet_size]

                #plt.text(x_loc, packet_size, f'(size={packet_size}, counts={packet_count})', c=COLORS['linux_tuned'])

            plt.ylabel('Transmitted Bytes')
            plt.title("Timeline plot for Transmitted Bytes")

        elif m=='rxbytes':
            plt.ylabel('Received Bytes')
            plt.title("Timeline plot for Received Bytes")

        if save_loc:
            plt.savefig(f'{save_loc}/nodejs_timeline_{m}.png')

    dfl['ins_per_ref_cycle'] = dfl['instructions_diff'] / dfl['refcyc_diff']
    dfl_tuned['ins_per_ref_cycle'] = dfl_tuned['instructions_diff'] / dfl_tuned['refcyc_diff']
    dfe['ins_per_ref_cycle'] = dfe['instructions_diff'] / dfe['refcyc_diff']

    #for m in ['joules_diff', 'llc_miss_diff', 'instructions_diff', 'ref_cycles_diff', 'ins_per_ref_cycle']:
    for m in ['joules_diff']:
        plt.figure()
        plt.plot(dfl['timestamp'], dfl[m], 'p', c=COLORS['linux_default'], label=LABELS['linux_default'])
        plt.plot(dfl_tuned['timestamp'], dfl_tuned[m], 'p', c=COLORS['linux_tuned'], label=LABELS['linux_tuned'])
        plt.plot(dfe['timestamp'], dfe[m], 'p', c=COLORS['ebbrt_tuned'], label=LABELS['ebbrt_tuned'])
        
        m = m.replace('_diff', '')

        if m=='ins_per_ref_cycle':
            continue

        d = dfl_orig[dfl_orig[m]==0]
        plt.plot(d['timestamp'], d[m], 'p', c=COLORS['linux_default'], marker='+')
        
        d = dfl_tuned_orig[dfl_tuned_orig[m]==0]
        plt.plot(d['timestamp'], d[m], 'p', c=COLORS['linux_tuned'], marker='+')
        
        d = dfe_orig[dfe_orig[m]==0]
        plt.plot(d['timestamp'], d[m], 'p', c=COLORS['ebbrt_tuned'], marker='+')

        plt.xlabel('Time (s)')
        
        plt.legend()
        plt.grid()
        
        plt.arrow(0.8, 0.018, 0.1, 0.005, width=0.0003)
        x = dfl[(dfl.joules_diff>0.0) & (dfl.joules_diff<0.02)]
        idle_power = x.joules_diff.sum() / (x.shape[0] * x['timestamp_diff'].mean())
        #plt.text(0.9 + 0.001, 0.023 + 0.001, f'Idle Power: {idle_power:.2f} Watts')

        if m=='joules':
            plt.ylabel('Energy Consumed (Joules)')
            plt.title('Timeline plot for Energy Consumed')
        
        if save_loc:
            plt.savefig(f'{save_loc}/nodejs_timeline_{m}.png')

        return dfl, dfl_orig, dfl_tuned, dfl_tuned_orig, dfe, dfe_orig 

def asplos_netpipe_tput_scan(save_loc):
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    def process_file(filename):
        df = pd.read_csv(filename, delim_whitespace=True, names=['msg', 'tput', 'x', 'y'])
        
        d1 = df.groupby('msg').mean()
        d1.columns = [f'{c}_mean' for c in d1.columns]
        
        d2 = df.groupby('msg').std()
        d2.columns = [f'{c}_std' for c in d2.columns]
        
        d = pd.concat([d1,d2],axis=1)
        d.reset_index(inplace=True)
        
        return d

    d_linux_default = process_file('aug11/linux_netpipe_itr_gov.csv')
    d_linux_tuned = process_file("aug11/linux_netpipe_itr_tuned.csv")
    d_ebbrt_tuned = process_file("aug11/ebbrt_scan.csv")

    plt.figure()
    plt.errorbar(d_linux_default.msg, d_linux_default.tput_mean, yerr=d_linux_default.tput_std, fmt='p-', c=COLORS['linux_default'], label=LABELS['linux_default'])
    plt.errorbar(d_linux_tuned.msg, d_linux_tuned.tput_mean, yerr=d_linux_tuned.tput_std, fmt='p-', c=COLORS['linux_tuned'], label=LABELS['linux_tuned'])
    plt.errorbar(d_ebbrt_tuned.msg, d_ebbrt_tuned.tput_mean, yerr=d_ebbrt_tuned.tput_std, fmt='p-', c=COLORS['ebbrt_tuned'], label=LABELS['ebbrt_tuned'])

def asplos_netpipe_edp_plots(save_loc, folder='aug16'):
    if not os.path.exists(save_loc):
        os.makedirs(save_loc)

    run_id = 2
    x_offset, y_offset = 0.01/5, 0.01/5

    df_comb, df, outlier_list = start_netpipe_analysis(drop_outliers=True)
    df_comb.reset_index(inplace=True)

    # These are in unitless percentages of the figure size. (0,0 is bottom left)
    for msg in [64, 8192, 65536, 524288]:
        fig = plt.figure(figsize=(9,7))

        #linux - default        
        filename = f'{folder}/governor/linux.dmesg.{run_id}_{msg}_5000_1_0xFFFF_135.csv'

        print(filename)
        if folder=='jul20':
            df = pd.read_csv(filename, sep = ' ', names=COLS)
        #elif folder=='aug19_netpipelogs':
        #    df = pd.read_csv(filename, sep = ' ', skiprows=1)
        #    df = rename_cols(df)
        else:
            df = pd.read_csv(filename, sep = ' ')

        J, T,_ = plot(df, LABELS['linux_default'], projection=True, color=COLORS['linux_default'], include_edp_label=True, JOULE_CONVERSION=JOULE_CONVERSION, TIME_CONVERSION=TIME_CONVERSION_khz)
        plt.text(x_offset + T, y_offset + J, f'(-, -, *)')

        #linux - tuned
        d = df_comb[(df_comb.sys=='linux') & (df_comb.msg==msg) & (df_comb.itr!=1) & (df_comb.dvfs!='0xFFFF')]
        row_linux_tuned = d[d.edp_mean==d.edp_mean.min()].copy()
        row = row_linux_tuned.iloc[0]

        assert(row.sys=='linux')

        filename = f'{folder}/rapl135/linux.dmesg.{run_id}_{msg}_5000_{row["itr"]}_{row["dvfs"]}_135.csv'

        print(filename)
        if folder=='jul20':
            df = pd.read_csv(filename, sep = ' ', names=COLS)
        #elif folder=='aug19_netpipelogs':
        #    df = pd.read_csv(filename, sep = ' ', skiprows=1)
        #    df = rename_cols(df)
        else:
            df = pd.read_csv(filename, sep = ' ')

        J,T,_ = plot(df, LABELS['linux_tuned'], projection=True, color=COLORS['linux_tuned'], include_edp_label=True, JOULE_CONVERSION=JOULE_CONVERSION, TIME_CONVERSION=TIME_CONVERSION_khz)
        plt.text(x_offset + T, y_offset + J, f'({row["itr"]}, {row["dvfs"]}, *)')

        #ebbrt - tuned
        d = df_comb[(df_comb.sys=='ebbrt') & (df_comb.msg==msg) & (df_comb.itr!=1) & (df_comb.dvfs!='0xFFFF')]
        row_ebbrt_tuned = d[d.edp_mean==d.edp_mean.min()].copy()
        row = row_ebbrt_tuned.iloc[0]

        assert(row.sys=='ebbrt')

        filename = f'{folder}/rapl135/ebbrt.dmesg.{run_id}_{msg}_5000_{row["itr"]}_{row["dvfs"]}_135.csv'
        print(filename)
        if folder=='jul20':
            df = pd.read_csv(filename, sep = ' ', names=COLS)
        else:
            df = pd.read_csv(filename, sep = ' ')
        J,T,_ = plot(df, LABELS['ebbrt_tuned'], projection=True, color=COLORS['ebbrt_tuned'], include_edp_label=True, JOULE_CONVERSION=JOULE_CONVERSION, TIME_CONVERSION=TIME_CONVERSION_khz)
        plt.text(x_offset + T, y_offset + J, f'({row["itr"]}, {row["dvfs"]}, *)')

        #axes etc.
        prettify()
        plt.title(f"Netpipe Experiment\n Msg Size = {msg} Bytes")

        plt.savefig(f'{save_loc}/netpipe_edp_{msg}.png')

    #inset plot
    #fig, ax1 = plt.subplots()
    #fig, (ax1, ax2) = plt.subplots(1, 2)
    fig = plt.figure(figsize=(9,7))
    #l,b,w,h = [0.65, 0.16, 0.22, 0.18]
    #ax2 = fig.add_axes([l,b,w,h])
    
    #plot 1: throughput vs msg size
    #1a: default linux
    d = df_comb[(df_comb['sys']=='linux') & (df_comb['itr']==1) & (df_comb['dvfs']=='0xFFFF')].copy()
    d.sort_values(by='msg', ascending=True, inplace=True)
    plt.errorbar(d['msg'], d['tput_mean'], yerr=d['tput_std'], fmt='p-', color=COLORS['linux_default'], label=LABELS['linux_default'])

    #1b: tuned linux - best for each message size
    y_linux_offset = 0.2
    d = df_comb[(df_comb['sys']=='linux') & (df_comb['itr']!=1) & (df_comb['dvfs']!='0xFFFF')].copy()
    #keep it simple
    msg_list, tput_list, tput_err_list, label_list = [], [], [], []
    for msg in [64, 8192, 65536, 524288]:
        e = d[d['msg']==msg]
        row_best_tput = e[e.tput_mean==e.tput_mean.max()].iloc[0]
        
        msg_list.append(row_best_tput['msg'])
        tput_list.append(row_best_tput['tput_mean'])
        tput_err_list.append(row_best_tput['tput_std'])
        label_list.append(f"({row_best_tput['itr']}, {row_best_tput['dvfs']}, *)")

    plt.errorbar(msg_list, tput_list, yerr=tput_err_list, fmt='p-', color=COLORS['linux_tuned'], label=LABELS['linux_tuned'])
    for idx, (msg, tput, label) in enumerate(zip(msg_list, tput_list, label_list)):
        plt.text(x_offset + msg, y_offset + y_linux_offset + tput, label, c=COLORS['linux_tuned'])

    #1c - ebbrt tuned
    y_ebbrt_offset = -0.1
    d = df_comb[(df_comb['sys']=='ebbrt') & (df_comb['itr']!=1) & (df_comb['dvfs']!='0xFFFF')].copy()
    #keep it simple
    msg_list, tput_list, tput_err_list, label_list = [], [], [], []
    for msg in [64, 8192, 65536, 524288]:
        e = d[d['msg']==msg]
        row_best_tput = e[e.tput_mean==e.tput_mean.max()].iloc[0]
        
        msg_list.append(row_best_tput['msg'])
        tput_list.append(row_best_tput['tput_mean'])
        tput_err_list.append(row_best_tput['tput_std'])
        label_list.append(f"({row_best_tput['itr']}, {row_best_tput['dvfs']}, *)")

    plt.errorbar(msg_list, tput_list, yerr=tput_err_list, fmt='p-', color=COLORS['ebbrt_tuned'], label=LABELS['ebbrt_tuned'])
    for idx, (msg, tput, label) in enumerate(zip(msg_list, tput_list, label_list)):
        plt.text(x_offset + msg, y_offset + y_ebbrt_offset + tput, label, c=COLORS['ebbrt_tuned'])

    plt.xlabel("Message Size (B)")
    plt.ylabel("Throughput (B/s)")
    plt.legend()
    plt.grid()
    plt.savefig(f'{save_loc}/netpipe_tput.png')

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

#nonidle time
#instructions
#joules
#bytes transmitted

def asplos_netpipe_exploratory_plots(msg_size = 8192, slice_middle=False, save_loc=None, folder='aug16'):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)


    dfl, dfl_tuned, dfe, dfl_orig, dfl_tuned_orig, dfe_orig = asplos_netpipe_log_analysis(msg_size=msg_size, save_loc=None, folder=folder, slice_middle=slice_middle)

    #for m in ['rx_bytes', 'rx_desc', 'tx_bytes', 'tx_desc']:
    for m in ['tx_bytes']:
        plt.figure()
        plt.plot(dfl_orig['timestamp'], dfl_orig[m], 'p', c=COLORS['linux_default'], label=LABELS['linux_default'])
        plt.plot(dfl_tuned_orig['timestamp'], dfl_tuned_orig[m], 'p', c=COLORS['linux_tuned'], label=LABELS['linux_tuned'])
        plt.plot(dfe_orig['timestamp'], dfe_orig[m], 'p', c=COLORS['ebbrt_tuned'], label=LABELS['ebbrt_tuned'])
        plt.xlabel('Time (s)')
        
        plt.legend()
        plt.grid()

        if m=='tx_bytes':
            counts = dfl_tuned_orig['tx_bytes'].value_counts()
            x_loc = dfl_tuned['timestamp'].max() * 1.1

            for packet_size in counts.index:
                packet_count = counts.loc[packet_size]

                #plt.text(x_loc, packet_size, f'(size={packet_size}, counts={packet_count})', c=COLORS['linux_tuned'])

            plt.ylabel('Transmitted Bytes')
            plt.title("Timeline plot for Transmitted Bytes")

        if save_loc:
            plt.savefig(f'{save_loc}/netpipe_timeline_{m}_{msg_size}.png')

    dfl['ins_per_ref_cycle'] = dfl['instructions_diff'] / dfl['ref_cycles_diff']
    dfl_tuned['ins_per_ref_cycle'] = dfl_tuned['instructions_diff'] / dfl_tuned['ref_cycles_diff']
    dfe['ins_per_ref_cycle'] = dfe['instructions_diff'] / dfe['ref_cycles_diff']

    #for m in ['joules_diff', 'llc_miss_diff', 'instructions_diff', 'ref_cycles_diff', 'ins_per_ref_cycle']:
    for m in ['joules_diff']:
        plt.figure()
        plt.plot(dfl['timestamp'], dfl[m], 'p', c=COLORS['linux_default'], label=LABELS['linux_default'])
        plt.plot(dfl_tuned['timestamp'], dfl_tuned[m], 'p', c=COLORS['linux_tuned'], label=LABELS['linux_tuned'])
        plt.plot(dfe['timestamp'], dfe[m], 'p', c=COLORS['ebbrt_tuned'], label=LABELS['ebbrt_tuned'])
        
        m = m.replace('_diff', '')

        if m=='ins_per_ref_cycle':
            continue

        d = dfl_orig[dfl_orig[m]==0]
        plt.plot(d['timestamp'], d[m], 'p', c=COLORS['linux_default'], marker='+')
        
        d = dfl_tuned_orig[dfl_tuned_orig[m]==0]
        plt.plot(d['timestamp'], d[m], 'p', c=COLORS['linux_tuned'], marker='+')
        
        d = dfe_orig[dfe_orig[m]==0]
        plt.plot(d['timestamp'], d[m], 'p', c=COLORS['ebbrt_tuned'], marker='+')

        plt.xlabel('Time (s)')
        
        plt.legend()
        plt.grid()
        
        plt.arrow(0.8, 0.018, 0.1, 0.005, width=0.0003)
        x = dfl[(dfl.joules_diff>0.0) & (dfl.joules_diff<0.02)]
        idle_power = x.joules_diff.sum() / (x.shape[0] * x['timestamp_diff'].mean())
        plt.text(0.9 + 0.001, 0.023 + 0.001, f'Idle Power: {idle_power:.2f} Watts')

        if m=='joules':
            plt.ylabel('Energy Consumed (Joules)')
            plt.title('Timeline plot for Energy Consumed')
        
        if save_loc:
            plt.savefig(f'{save_loc}/netpipe_timeline_{m}_{msg_size}.png')



def asplos_netpipe_log_analysis(msg_size = 65536, save_loc=None, folder='aug16', slice_middle=True):
    if save_loc is not None:
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

    run_id = 2

    #prepare data
    df_comb, df, outlier_list = start_netpipe_analysis(drop_outliers=True)
    df_comb.reset_index(inplace=True)

    df_comb = df_comb[df_comb.msg==msg_size]

    #linux default
    filename = f'{folder}/governor/linux.dmesg.{run_id}_{msg_size}_5000_1_0xFFFF_135.csv'
    dfl, dfl_orig = asplos_log_plots(filename, folder, slice_middle=slice_middle)
    print(filename)

    #linux tuned
    d = df_comb[(df_comb['sys']=='linux') & (df_comb['itr']!=1) & (df_comb['dvfs']!='0xFFFF')].copy()
    row_best_edp = d[d.edp_mean==d.edp_mean.min()].iloc[0]

    filename = f'{folder}/rapl135/linux.dmesg.{run_id}_{msg_size}_5000_{row_best_edp["itr"]}_{row_best_edp["dvfs"]}_135.csv'
    dfl_tuned, dfl_tuned_orig = asplos_log_plots(filename, folder, slice_middle=slice_middle)
    print(filename)

    #ebbrt tuned
    d = df_comb[(df_comb['sys']=='ebbrt') & (df_comb['itr']!=1) & (df_comb['dvfs']!='0xFFFF')].copy()
    row_best_edp = d[d.edp_mean==d.edp_mean.min()].iloc[0]

    filename = f'{folder}/rapl135/ebbrt.dmesg.{run_id}_{msg_size}_5000_{row_best_edp["itr"]}_{row_best_edp["dvfs"]}_135.csv'
    dfe, dfe_orig = asplos_log_plots(filename, folder, slice_middle=slice_middle)
    print(filename)

    #busy plot
    plt.figure(figsize=(9,7))
    plt.plot(dfl['timestamp'], dfl['nonidle_frac_diff'], 'p-', label=LABELS['linux_default'])
    plt.plot(dfl_tuned['timestamp'], dfl_tuned['nonidle_frac_diff'], 'p-', label=LABELS['linux_tuned'], c=COLORS['linux_tuned'])
    plt.plot(dfe['timestamp'], dfe['nonidle_frac_diff'], 'p-', label=LABELS['ebbrt_tuned'], c=COLORS['ebbrt_tuned'])
    
    plt.xlabel('Time (s)')
    plt.ylabel('Non-idle Time (%)')
    plt.ylim((0, 1.001))
    if slice_middle:
        plt.title(f"Non-idle Fractional Time in 0.2sec window for Msg Size = {msg_size} Bytes")
    else:
        plt.title(f"Non-idle Fractional Time for Msg Size = {msg_size} Bytes")

    plt.grid()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), 
               loc='lower left', 
               #mode="expand", 
               borderaxespad=0.)
    
    if save_loc is not None:
        if slice_middle: 
            plt.savefig(f'{save_loc}/netpipe_nonidle_{msg_size}_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/netpipe_nonidle_{msg_size}.png')

    #instructions
    plt.figure()
    plt.bar([1,2,3], [dfl['instructions'].tail(1).iloc[0], \
                      dfl_tuned['instructions'].tail(1).iloc[0], \
                      dfe['instructions'].tail(1).iloc[0]],
                      color = [COLORS['linux_default'], COLORS['linux_tuned'], COLORS['ebbrt_tuned']])
    plt.xticks(ticks=[1,2,3], labels = [LABELS['linux_default'], LABELS['linux_tuned'], LABELS['ebbrt_tuned']])
    plt.ylabel('Total Instructions')
    if slice_middle:
        plt.title(f"Total Instructions in 0.2sec window from 1 sample run\n Msg Size = {msg_size} Bytes")
    else:
        plt.title(f"Total Instructions from 1 sample run\n Msg Size = {msg_size} Bytes")

    if save_loc is not None:
        if slice_middle:
            plt.savefig(f'{save_loc}/netpipe_instructions_{msg_size}_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/netpipe_instructions_{msg_size}.png')

    #joules
    plt.figure()
    plt.bar([1,2,3], [dfl['joules'].tail(1).iloc[0], \
                      dfl_tuned['joules'].tail(1).iloc[0], \
                      dfe['joules'].tail(1).iloc[0]],
                      color = [COLORS['linux_default'], COLORS['linux_tuned'], COLORS['ebbrt_tuned']])
    plt.xticks(ticks=[1,2,3], labels = [LABELS['linux_default'], LABELS['linux_tuned'], LABELS['ebbrt_tuned']])
    plt.ylabel('Energy Used (Joules)')
    if slice_middle:
        plt.title(f"Energy Consumed in 0.2sec window from 1 sample run\n Msg Size = {msg_size} Bytes")
    else:
        plt.title(f"Total Energy Consumed from 1 sample run\n Msg Size = {msg_size} Bytes")
    
    if save_loc is not None:
        if slice_middle:
            plt.savefig(f'{save_loc}/netpipe_energy_{msg_size}_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/netpipe_energy_{msg_size}.png')

    #bytes transmitted
    plt.figure()
    plt.bar([1,2,3], [dfl_orig['tx_bytes'].sum(), \
                      dfl_tuned_orig['tx_bytes'].sum(), \
                      dfe_orig['tx_bytes'].sum()],
                      color = [COLORS['linux_default'], COLORS['linux_tuned'], COLORS['ebbrt_tuned']])
    plt.xticks(ticks=[1,2,3], labels = [LABELS['linux_default'], LABELS['linux_tuned'], LABELS['ebbrt_tuned']])
    plt.ylabel('Data Transmitted (Bytes)')
    if slice_middle:
        plt.title(f"Data Transmitted in 0.2sec window from 1 sample run\n Msg Size = {msg_size} Bytes")
    else:
        plt.title(f"Total Data Transmitted from 1 sample run\n Msg Size = {msg_size} Bytes")

    if save_loc is not None:
        if slice_middle:
            plt.savefig(f'{save_loc}/netpipe_txbytes_{msg_size}_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/netpipe_txbytes_{msg_size}.png')

    #transmission efficiency
    plt.figure()
    plt.bar([1,2,3], [dfl_orig['tx_bytes'].sum() / dfl['joules'].tail(1).iloc[0], \
                      dfl_tuned_orig['tx_bytes'].sum() / dfl_tuned['joules'].tail(1).iloc[0], \
                      dfe_orig['tx_bytes'].sum() / dfe['joules'].tail(1).iloc[0]],
                      color = [COLORS['linux_default'], COLORS['linux_tuned'], COLORS['ebbrt_tuned']])
    plt.xticks(ticks=[1,2,3], labels = [LABELS['linux_default'], LABELS['linux_tuned'], LABELS['ebbrt_tuned']])
    plt.ylabel('Data Transmitted / Energy (Bytes / Joule)')
    if slice_middle:
        plt.title(f"Data Transmission Efficiency in 0.2sec window from 1 sample run\n Msg Size = {msg_size} Bytes")
    else:
        plt.title(f"Data Transmission Efficiency from 1 sample run\n Msg Size = {msg_size} Bytes")

    if save_loc is not None:
        if slice_middle:
            plt.savefig(f'{save_loc}/netpipe_instructionsperjoule_{msg_size}_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/netpipe_instructionsperjoule_{msg_size}.png')

    #cluster all bars together
    metric_labels = ['Instructions', 'Energy', 'Cycles', 'Data Transmitted', 'Bytes/Joule', 'Interrupts']
    N_metrics = len(metric_labels) #number of clusters
    N_systems = 3 #number of plot loops

    fig = plt.figure()
    ax = fig.subplots()

    idx = np.arange(N_metrics) #one group per metric
    width = 0.2

    df_dict = {'linux_default': (dfl, dfl_orig),
               'linux_tuned': (dfl_tuned, dfl_tuned_orig),
               'ebbrt_tuned': (dfe, dfe_orig)}
    data_dict = {}

    for sys in df_dict: #compute metrics
        data_dict[sys] = np.array([df_dict[sys][0]['instructions'].tail(1).iloc[0],
                                   df_dict[sys][0]['joules'].tail(1).iloc[0],
                                   df_dict[sys][0]['ref_cycles'].tail(1).iloc[0],
                                   df_dict[sys][1]['tx_bytes'].sum(),
                                   df_dict[sys][1]['tx_bytes'].sum() / df_dict[sys][0]['joules'].tail(1).iloc[0],
                                   df_dict[sys][1].shape[0]
                                  ])

    counter = 0
    for sys in data_dict: #normalize and plot
        data = data_dict[sys] / data_dict['linux_default']

        ax.bar(idx + counter*width, data, width, label=LABELS[sys], color=COLORS[sys])
        counter += 1

    ax.set_xticks(idx)
    ax.set_xticklabels(metric_labels, rotation=15, fontsize='small')
    ax.set_ylabel('Metric / Metric for Linux Default')
    plt.legend()

    if slice_middle:
        plt.title(f"Relative Metrics in 0.2sec window from 1 sample run\n Msg Size = {msg_size} Bytes")
    else:
        plt.title(f"Relative Metrics from 1 sample run\n Msg Size = {msg_size} Bytes")

    if save_loc is not None:
        if slice_middle:
            plt.savefig(f'{save_loc}/netpipe_combined_barplot_{msg_size}_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/netpipe_combined_barplot_{msg_size}.png')



    #stacked energy plot

    '''
    3 systems
    total energy -> dfl, dfl_tuned, dfe
    busy time -> see non-idle-frac
    energy calc: idle_perc * total_time * base_power
    stacking: levels -> 1. total/realized, 2. idle power = 0
    '''

    #stacked interrupt delays

    N_systems = 3 #number of plot loops

    fig = plt.figure()
    ax = fig.subplots()

    idx = np.arange(N_systems) #one group per metric
    width = 0.2

    df_dict = {'linux_default': (dfl, dfl_orig),
               'linux_tuned': (dfl_tuned, dfl_tuned_orig),
               'ebbrt_tuned': (dfe, dfe_orig)}
    data_dict = {}

    keys = list(df_dict.keys())

    power_idle = {'linux_default': 17,
                  'linux_tuned': 17,
                  'ebbrt_tuned': 17}

    joules = np.array([df_dict[sys][0]['joules'].tail(1).iloc[0] for sys in keys])
    
    joules_idle = []
    for sys in keys:
        total_time = df_dict[sys][0]['timestamp'].tail(1).iloc[0]
        
        idle_perc = 1 - df_dict[sys][0]['nonidle_frac_diff'].mean()

        d = total_time * idle_perc * power_idle[sys]

        joules_idle.append(d)

    joules_idle = np.array(joules_idle)

    ax.bar(idx, joules_idle, width, label='Estimated Idling Energy Consumption')
    plt.plot(idx, joules, 'p', marker='+', label='Measured Total Energy Consumption')
    ax.bar(idx, joules-joules_idle, width, bottom=joules_idle, label='Measured Minus Idle Estimate', alpha=0.2)

    ax.set_xticks(idx)
    labels = [LABELS[l] for l in keys]
    ax.set_xticklabels(labels, rotation=0, fontsize='small')
    ax.set_ylabel("Joules")
    plt.title('Energy Consumption Breakdown')
    plt.legend()

    if slice_middle:
        plt.title(f"Energy breakdown by time in 0.2sec window from 1 sample run\n Msg Size = {msg_size} Bytes")
    else:
        plt.title(f"Energy breakdown by time from 1 sample run\n Msg Size = {msg_size} Bytes")

    if save_loc is not None:
        if slice_middle:
            plt.savefig(f'{save_loc}/netpipe_energy_breakdown_{msg_size}_window200ms.png')
        else:
            plt.savefig(f'{save_loc}/netpipe_energy_breakdown_{msg_size}.png')


    return dfl, dfl_tuned, dfe, dfl_orig, dfl_tuned_orig, dfe_orig

def asplos_plots():
    #netpipe
    df_comb, df, outlier_list = start_netpipe_analysis(drop_outliers=True) 

    df_comb.reset_index(inplace=True)

    #plot 1: throughput vs msg size
    plt.figure()
    #1a: default linux
    d = df_comb[(df_comb['sys']=='linux') & (df_comb['itr']==1) & (df_comb['dvfs']=='0xFFFF')].copy()
    d.sort_values(by='msg', ascending=True, inplace=True)
    plt.errorbar(d['msg'], d['tput_mean'], yerr=d['tput_std'], fmt='p-', color='orange', label=LABELS['linux_default'])

    #1b: tuned linux
    max_tput = df_comb[(df_comb['sys']=='linux') & (df_comb['itr']!=1) & (df_comb['dvfs']!='0xFFFF')].groupby(['sys', 'msg']).max()['tput_mean']['linux'].to_dict()
    d = df_comb[df_comb.apply(lambda x: x['tput_mean']==max_tput[x['msg']], axis=1)].copy()
    d.sort_values(by='msg', ascending=True, inplace=True)
    plt.errorbar(d['msg'], d['tput_mean'], yerr=d['tput_std'], fmt='p-', color='green', label=LABELS['linux_tuned'])

    #1c: tuned ebbrt
    max_tput = df_comb[(df_comb['sys']=='ebbrt') & (df_comb['itr']!=1) & (df_comb['dvfs']!='0xFFFF')].groupby(['sys', 'msg']).max()['tput_mean']['ebbrt'].to_dict()
    d = df_comb[df_comb.apply(lambda x: x['tput_mean']==max_tput[x['msg']], axis=1)].copy()
    d.sort_values(by='msg', ascending=True, inplace=True)
    plt.errorbar(d['msg'], d['tput_mean'], yerr=d['tput_std'], fmt='p-', color='red', label=LABELS['ebbrt_tuned'])

    plt.xlabel("Message Size (B)")
    plt.ylabel("Throughput (B/s)")

def read_all_csv(foldername, 
                 label,
                 df = {}, 
                 extension='.csv',
                 column_names=None,
                 skiprows=None):
    #df = {}
    for file in os.listdir(foldername):
        if file.find(extension)>-1:
            if column_names is None:
                df[(label, file)] = pd.read_csv(os.path.join(foldername, file), 
                                                sep = ' ',
                                                error_bad_lines=False,
                                                warn_bad_lines=True,
                                                skiprows=skiprows)
            else:
                df[(label, file)] = pd.read_csv(os.path.join(foldername, file), 
                                                header = 0,
                                                names = column_names,
                                                sep = ' ',
                                                error_bad_lines=False,
                                                warn_bad_lines=True,
                                                skiprows=skiprows)
    return df


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

def plot(df, 
         label, 
         projection=False, 
         color='orange', 
         plot_every=50,
         include_edp_label=False,
         JOULE_CONVERSION=1,
         TIME_CONVERSION=1):

    df_non0j = df[df['joules']>0].copy()

    #cleanup
    drop_cols = [c for c in df_non0j.columns if c.find('Unnamed')>-1]
    if len(drop_cols) > 0: df_non0j.drop(drop_cols, axis=1, inplace=True)
    print(f'Dropping null rows: {df_non0j.shape[0] - df_non0j.dropna().shape[0]} rows')
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
    
    print(f'EDP               : {edp_val}')
    #print(f'Total Instructions: {df_non0j["instructions"].sum()}')
    #print(f'Total Cache Misses: {df_non0j["llc_miss"].sum()}')
    #print(f'Total Cycles      : {df_non0j["cycles"].sum()}')

    if include_edp_label: label = f'{label} EDP={edp_val:.2f}'

    #plot line and points
    plt.plot(df_non0j['timestamp'], df_non0j['joules'], '-', color = color)
    plt.plot(df_non0j['timestamp'].iloc[::plot_every], df_non0j['joules'].iloc[::plot_every], 'p', label=label, color=color)    

    #plot lines projecting onto the x and y axes from the end-point
    if projection:
        plt.plot([df_non0j['timestamp'].max(), df_non0j['timestamp'].max()], 
                 [df_non0j['joules'].max(), 0], '--', color=color)

        plt.plot([df_non0j['timestamp'].max(), 0], 
                 [df_non0j['joules'].max(), df_non0j['joules'].max()], '--', color=color)

    return last_row['joules'], last_row['timestamp'], df_non0j

def prettify(xlabel = 'Time (seconds)', 
             ylabel = 'Energy Consumed (Joules)', 
             prop={},
             ncol=1):
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xlim(0, plt.xlim()[1])
    plt.ylim(0, plt.ylim()[1]) 

    plt.grid()
    plt.legend()
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), 
               loc='lower left', 
               ncol=ncol, 
               #mode="expand", 
               borderaxespad=0.,
               prop=prop)

def clean_name(k):
    return k.replace('.csv','').replace('.stats', '').replace('base_linux_', '')

def make_abstract_plot():
    XMAX = 10

    fig = plt.figure()

    points = ([0.8, np.sqrt(9*0.8 / (5.3/4.3)), 4.3, 9.], [9, 9*0.8/np.sqrt(9*0.8 / (5.3/4.3)), 5.3, 3.5])
    EDP_vals = [x*y for (x,y) in zip(*points)]
    colors = ['orange', 'orange', 'green', 'blue']

    offset = 0.1
    #plot points with EDP areas
    for idx in range(len(points[0])):
        plt.plot([0, points[0][idx]], [0, points[1][idx]], 'p-', color=colors[idx], markerfacecolor=colors[idx])
        plt.plot([points[0][idx], points[0][idx]], [0, points[1][idx]], '--', color=colors[idx])
        plt.plot([0, points[0][idx]], [points[1][idx], points[1][idx]], '--', color=colors[idx])

        #plot points on line
        x_line = np.linspace(0, points[0][idx], 5)
        y_line = points[1][idx] / points[0][idx] * x_line
        plt.plot(x_line, y_line, 'p', color=colors[idx])

        plt.text(points[0][idx] + offset, points[1][idx] + offset, chr(65+idx))

        ax = fig.axes[0]

        x_range = [0, points[0][idx]]
        y_low = [0, 0]
        y_high = [0, points[1][idx]]
        ax.fill_between(x_range, y_low, y_high, color=colors[idx], alpha=0.3)

    #plot EDP curves
    for idx in range(1, len(EDP_vals)):
        EDP = EDP_vals[idx]

        x_vals = np.linspace(0.1, XMAX, 50)
        y_vals = EDP / x_vals
        plt.plot(x_vals, y_vals, '-', label=f'EDP = {EDP:.2f}', color=colors[idx])

        #plot 3 random points
        #XMIN = EDP / XMAX
        #x_rnd = np.random.choice(np.linspace(XMIN, XMAX, 50), 3)
        #y_rnd = EDP / x_rnd
        #plt.plot(x_rnd, y_rnd, 'p', color=colors[idx])

    plt.grid()
    plt.legend()

    plt.xlim([0, XMAX])
    plt.ylim([0, XMAX])
    plt.xlabel("Time Taken (arbitrary units)")
    plt.ylabel("Energy Consumed (arbitrary units)")
    plt.title("Energy vs Time with Curves of \nConstant Energy-Delay Product Values")

