import os
import matplotlib.pylab as plt
from utils import *
from constants import *

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
    