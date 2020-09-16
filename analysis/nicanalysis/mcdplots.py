import os
import matplotlib.pylab as plt
from utils import *
from constants import *

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
    