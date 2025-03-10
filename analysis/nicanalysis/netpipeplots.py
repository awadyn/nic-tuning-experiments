import os
import matplotlib.pylab as plt
import matplotlib as mpl
import numpy as np
from utils import *
from constants import *

plt.ion()
plt.rcParams.update({'font.size': 14})
pd.set_option('display.max_rows', 50)

#frame.number -e frame.time_epoch -e frame.time_delta_displayed -e ip.src -e ip.dst -e tcp.port -e frame.cap_len -e tcp.window_size -e tcp.analysis.fast_retransmission -e tcp.analysis.retransmission
#i rx_desc rx_bytes tx_desc tx_bytes instructions cycles ref_cycles llc_miss c3 c6 c7 joules timestamp
COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c3', 'c6', 'c7', 'joules', 'timestamp']
PCAP_COLS = ['frame_number', 'frame_time_epoch', 'frame_time_delta', 'ip_src', 'ip_dst', 'port_src', 'port_dst', 'frame_len', 'tcp_window_size', 'tcp_fast_retransmission', 'tcp_retransmission']

def prepare_scan_all_data(df):
    #COLS = ['frame_time_epoch', 'frame_time_delta', 'frame_len', 'tcp_window_size', 'tcp_fast_retransmission', 'tcp_retransmission']
    COLS = ['frame_number', 'ip_src', 'ip_dst', 'port_src', 'port_dst']

    df_mean = df.groupby(COLS).mean()
    df_std = df.groupby(COLS).std()
    
    df_mean.columns = [f'{c}_mean' for c in df_mean.columns]
    df_std.columns = [f'{c}_std' for c in df_std.columns]

    df_comb = pd.concat([df_mean, df_std], axis=1)

    return df_comb

def netpipe_pcap_tmp_plot(folder='', run_id=0, msgs='8192', core=1, dvfs='0x1900', rapl=135, itrs='2', xax='frame_number', yax='frame_time_delta'):
    for itr in itrs.split(' '):
        filename = f'{folder}/tshark.pcap.{run_id}_{core}_{msgs}_5000_{itr}_{dvfs}_135.csv'
        df = pd.read_csv(filename, sep=',', names=PCAP_COLS, skiprows=500, skipfooter=500)
        #df['frame_time_delta'] = df['frame_time_delta'].cumsum()
        fig, ax = plt.subplots()
        ax.ticklabel_format(style='plain')
        plt.title(f'MSG={msgs} ITR={itr}')
        plt.plot(df[xax], df[yax], '*', c='r')
        plt.legend()
        plt.grid()
        plt.xlabel(xax)
        plt.ylabel(yax)
        plt.ylim(bottom=0)
        
        
        '''
        # server -> client
        df_sc = df[df['ip_dst'] == '192.168.1.11'].copy()
        fig, ax = plt.subplots()
        plt.title(f'MSG={msgs} ITR={itr} \n 192.168.1.9->192.168.1.11')
        plt.plot(df_sc[xax], df_sc[yax], '*', c='r')
        plt.legend()
        plt.grid()
        plt.xlabel(xax)
        plt.ylabel(yax)
        plt.ylim(bottom=0)
        
        # client->server
        plt.figure()
        df_cs = df[df['ip_dst'] == '192.168.1.9'].copy()
        plt.title(f'MSG={msgs} ITR={itr} \n 192.168.1.11->192.168.1.9')
        plt.plot(df_cs[xax], df_cs[yax], '*', c='b')
        plt.legend()
        plt.grid()
        plt.xlabel(xax)
        plt.ylabel(yax)
        plt.ylim(bottom=0)        
        '''
        
def netpipe_pcap_itr_plots(folder='', run_id=0, msgs='8192', core=1, dvfs='0x1d00', rapl=135, itrs='2', yax='frame_time_delta'):
    for msg in msgs.split(' '):
        ty=[]
        for itr in itrs.split(' '):
            filename = f'{folder}/tshark.pcap.{run_id}_{core}_{msg}_5000_{itr}_{dvfs}_135.csv'
            df = pd.read_csv(filename, sep=',', skiprows=500, skipfooter=500, names=PCAP_COLS)
            df = df[df['ip_dst'] == '192.168.1.11']
            ty.append(df[yax].mean())
        fig, ax = plt.subplots()
        #plt.title(f'RUN_ID={run_id} | MSGS={msgs}')
        print(itrs, ty)
        ax.ticklabel_format(style='plain')
        ax.plot(itrs.split(' '), ty, 'x')         
        plt.legend()
        plt.grid()
        plt.xlabel('ITR')
        plt.ylabel(yax)

def netpipe_pcap_itr_time_diff(folder='', run_id=0, msgs='8192', core=1, dvfs='0x1d00', rapl=135, itrs='2', plot_default=False):
    time_diff=[]
    time_diff_std=[]
    yl = []
    yu = []
    for dv in dvfs.split(' '):
        plt.figure()
        for msg in msgs.split(' '):
            for itr in itrs.split(' '):
                filename = f'{folder}/tshark.pcap.{run_id}_{core}_{msg}_5000_{itr}_{dv}_135.csv'
                df = pd.read_csv(filename, sep = ',', names=PCAP_COLS)
                #frame_time_delta
                #df['timestamp_diff'] = df['timestamp'].diff()

                ## drop first and last 100 rows
                df.drop(df.tail(100).index,inplace=True)
                df.drop(df.head(100).index,inplace=True)
                
                df_timestamp_diff_mean = df['frame_time_delta'].mean() * 1000000.0
                time_diff.append(df_timestamp_diff_mean)
                
                df_max = df['frame_time_delta'].max() * 1000000.0
                df_min = df['frame_time_delta'].min() * 1000000.0                                
                #df_timestamp_diff_std = df['timestamp_diff'].std() * 1000000.0                
                #time_diff_std.append(df_timestamp_diff_std)
                yl.append(df_timestamp_diff_mean-df_min)
                yu.append(df_max-df_timestamp_diff_mean)
                #time_diff_std.append([, )
                #time_diff_std = np.transpose(time_diff_std)
                
            #plt.errorbar(x=itrs.split(' '), y=time_diff, yerr=time_diff_std, fmt='*', label=f'MSG={msg}')
            plt.errorbar(x=itrs.split(' '), y=time_diff, yerr=[yl, yu], fmt='*', label=f'MSG={msg}')            
            time_diff=[]
            time_diff_std=[]
            yl=[]
            yu=[]
            plt.xlabel('ITR-Delay')
        plt.ylabel('mean frame_time_delta (us)')
        plt.legend()
        plt.title('Wireshark yerr=[mean-min, max-mean]')
        plt.ylim(top=300)
        plt.grid()

        
def netpipe_pcap_plots(folder='', run_id=0, msgs='8192', core=1, dvfs='0x1d00', rapl=135, itrs='2', yax='frame_time_delta'):
    itr_min = int(itrs.split(' ')[0])
    itr_max = int(itrs.split(' ')[len(itrs.split(' '))-1])
    msg_min = int(msgs.split(' ')[0])
    msg_max = int(msgs.split(' ')[len(msgs.split(' '))-1])
    
    for msg in msgs.split(' '):
        fig, ax = plt.subplots()
        plt.title(f'RUN_ID={run_id} | MSGS={msgs} | ITRS={itr_min, itr_max}')        
        for itr in itrs.split(' '):
            filename = f'{folder}/tshark.pcap.{run_id}_{core}_{msg}_5000_{itr}_{dvfs}_135.csv'
            print(f'Reading {filename}')
            df = pd.read_csv(filename, sep=',', skiprows=500, skipfooter=500, names=PCAP_COLS)
            df = df[df['ip_dst'] == '192.168.1.11']
            ax.plot(df['frame_number'], df[yax], 'x', alpha=0.5, label=f'ITR={itr}')
        ax.ticklabel_format(style='plain')
        plt.legend()
        plt.grid()
        plt.xlabel('frame_number')
        plt.ylabel('frame_time_delta')
                                           
def netpipe_random_plots(folder='', run_id=0, msgs='8192', core=1, dvfs='0x1d00', rapl=135, itrs='2', plot_default=False):
    num_interrupts=[]
    plt.figure()
    itr_min = int(itrs.split(' ')[0])
    itr_max = int(itrs.split(' ')[len(itrs.split(' '))-1])

    msg_min = int(msgs.split(' ')[0])
    msg_max = int(msgs.split(' ')[len(msgs.split(' '))-1])
    
    plt.title(f'RUN_ID={run_id} | MSGS={msg_min}...{msg_max} | ITRS={itr_min}...{itr_max}')
    if plot_default:
        for msg in msgs.split(' '):
            filename = f'{folder}/linux.np.log.{run_id}_{core}_{msg}_5000_1_0xFFFF_135.csv'
            df = pd.read_csv(filename, sep = ' ')
            num_interrupts.append(df.shape[0])
        plt.plot(msgs.split(' '), num_interrupts, 'x-', label='ITR=Default', color='red')
        num_interrupts=[]
        
    norm = mpl.colors.Normalize(vmin=itr_min, vmax=itr_max)
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])

    for itr in itrs.split(' '):
        for msg in msgs.split(' '):        
            filename = f'{folder}/linux.np.log.{run_id}_{core}_{msg}_5000_{itr}_{dvfs}_135.csv'
            df = pd.read_csv(filename, sep = ' ')
            num_interrupts.append(df.shape[0])    
        plt.plot(msgs.split(' '), num_interrupts, '*-', label='ITR='+itr, c=cmap.to_rgba(int(itr) + 1))
        num_interrupts=[]    
    plt.xlabel('MSG_SIZE (Bytes)')
    plt.ylabel('NUM_INTERRUPTS')
    plt.legend()
    plt.grid()

def get_edp(df):
    df_non0j = df[df['joules']>0].copy()
    df_non0j['timestamp'] = df_non0j['timestamp'] - df_non0j['timestamp'].min()
    df_non0j['joules'] = df_non0j['joules'] - df_non0j['joules'].min()
    
    df_non0j['timestamp'] = df_non0j['timestamp'] * TIME_CONVERSION_khz
    df_non0j['joules'] = df_non0j['joules'] * JOULE_CONVERSION
    
    #get edp value
    last_row = df_non0j.tail(1).iloc[0]
    edp_val = 0.5 * last_row['joules'] * last_row['timestamp']
    return edp_val
                
def netpipe_edp_compare(folder1='', folder2='', save_loc='', run_id=0, msgs='8192', core=1, dvfs='0x1900', rapl=135, itrs='2'):
    edp1=[]
    edp2=[]
    
    itr_min = int(itrs.split(' ')[0])
    itr_max = int(itrs.split(' ')[len(itrs.split(' '))-1])

    for msg in msgs.split(' '):
        for dv in dvfs.split(' '):
            for itr in itrs.split(' '):
                filename1 = f'{folder1}/linux.np.log.{run_id}_{core}_{msg}_5000_{itr}_{dv}_135.csv'
                df = pd.read_csv(filename1, sep = ' ')
                edp1.append(get_edp(df))
                
                filename2 = f'{folder2}/linux.np.log.{run_id}_{core}_{msg}_5000_{itr}_{dv}_135.csv'
                df2 = pd.read_csv(filename2, sep = ' ')
                edp2.append(get_edp(df2))

            plt.figure()
            plt.title(f'RUN_ID={run_id} | MSGS={msg} | ITRS={itr_min}-{itr_max}')
            plt.plot(itrs.split(' '), edp1, '*', label=f'{folder1}')
            plt.plot(itrs.split(' '), edp2, '*', label=f'{folder2}')
            plt.xlabel('ITR-Delay')
            plt.ylabel('EDP')
            plt.ylim(bottom=0)
            plt.legend()
            plt.grid()
            if save_loc:
                plt.savefig(f'{save_loc}/netpipe_edp_compare_{msg}.png')
            edp1=[]
            edp2=[]

def netpipe_itr_time_diff(folder='', run_id=0, msgs='8192', core=1, dvfs='0x1d00', rapl=135, itrs='2', plot_default=False):
    time_diff=[]
    time_diff_std=[]
    yl = []
    yu = []
    for dv in dvfs.split(' '):
        plt.figure()
        for msg in msgs.split(' '):
            for itr in itrs.split(' '):
                filename = f'{folder}/linux.np.log.{run_id}_{core}_{msg}_5000_{itr}_{dv}_135.csv'
                df = pd.read_csv(filename, sep = ' ')
                df['timestamp'] = df['timestamp'] * TIME_CONVERSION_khz
                df['timestamp_diff'] = df['timestamp'].diff()

                ## drop first and last 100 rows
                df.drop(df.tail(100).index,inplace=True)
                df.drop(df.head(100).index,inplace=True)
                
                df_timestamp_diff_mean = df['timestamp_diff'].mean() * 1000000.0
                time_diff.append(df_timestamp_diff_mean)
                
                df_max = df['timestamp_diff'].max() * 1000000.0
                df_min = df['timestamp_diff'].min() * 1000000.0                                
                df_timestamp_diff_std = df['timestamp_diff'].std() * 1000000.0                
                time_diff_std.append(df_timestamp_diff_std)
                #yl.append(df_timestamp_diff_mean-df_min)
                #yu.append(df_max-df_timestamp_diff_mean)
                
                
            plt.errorbar(x=itrs.split(' '), y=time_diff, yerr=time_diff_std, fmt='*', label=f'MSG={msg}')
            #plt.errorbar(x=itrs.split(' '), y=time_diff, yerr=[yl, yu], fmt='*', label=f'MSG={msg}')            
            time_diff=[]
            time_diff_std=[]
            yl=[]
            yu=[]
            plt.xlabel('ITR-Delay')
        plt.ylabel('mean timestamp diff (us)')
        plt.title('yerr=[mean-min, max-mean]')
        plt.legend()
        plt.grid()
        #plt.ylim(top=300)                

def netpipe_lowitr(folder='', run_id=0, msgs='8192', core=1, dvfs='0x1d00', rapl=135, itrs='2', plot_default=False):
    mitrs =[]
    for msg in msgs.split(' '):
        for dv in dvfs.split(' '):
            min_itr=9999999
            for itr in itrs.split(' '):
                filename = f'{folder}/linux.np.log.{run_id}_{core}_{msg}_5000_{itr}_{dv}_135.csv'
                df = pd.read_csv(filename, sep = ' ')
                if df.shape[0] < min_itr:
                    min_itr = df.shape[0]
            mitrs.append(min_itr)
    plt.figure()
    #plt.title(f'RUN_ID={run_id} | MSGS={msg} | ITRS={itr_min}...{itr_max}')
    plt.plot(msgs.split(' '), mitrs, '*')
    plt.xlabel('Message Size')
    plt.ylabel('Min Num ITRS')
    plt.grid()
                

            
def netpipe_entropy(folder='', run_id=0, msgs='8192', core=1, dvfs='0x1d00', rapl=135, itrs='2', plot_default=False):
    enl=[]
    edpl=[]
    ovrhl=[]
    nitr=[]
    tputl=[]
    itr_min = int(itrs.split(' ')[0])
    itr_max = int(itrs.split(' ')[len(itrs.split(' '))-1])

    msg_min = int(msgs.split(' ')[0])
    msg_max = int(msgs.split(' ')[len(msgs.split(' '))-1])
        
    for msg in msgs.split(' '):
        ## print Linux Default
        if plot_default:
            filename = f'{folder}/linux.np.log.{run_id}_{core}_{msg}_5000_1_0xFFFF_135.csv'
            #print(filename)
            df = pd.read_csv(filename, sep = ' ')
            d = df['rx_bytes'].value_counts()

            ## filter out 0's
            d = d[d.index > 0]
            sumt = df['rx_bytes'].sum()
            total = d.sum()
            d = d/total
            entropy = -(d * np.log(d)).sum()
            enl.append(entropy)
            print(f'MSG={msg} ITR=1 RX_BYTE_OVERHEAD={(sumt/(float(msg)*5000)) - 1.0} NUM_INTERRUPTS={df.shape[0]} ENTROPY={entropy}')
            ovrhl.append((sumt/(float(msg)*5000)) - 1.0)
            nitr.append(df.shape[0])
            df_non0j = df[df['joules']>0].copy()
            df_non0j['timestamp'] = df_non0j['timestamp'] - df_non0j['timestamp'].min()
            df_non0j['joules'] = df_non0j['joules'] - df_non0j['joules'].min()
    
            df_non0j['timestamp'] = df_non0j['timestamp'] * TIME_CONVERSION_khz
            df_non0j['joules'] = df_non0j['joules'] * JOULE_CONVERSION                
                
            #get edp value
            last_row = df_non0j.tail(1).iloc[0]
            edp_val = 0.5 * last_row['joules'] * last_row['timestamp']
            edpl.append(edp_val)
            tputl.append(int(msg)*5000 / float(last_row['timestamp']))         
                
        for dv in dvfs.split(' '):
            for itr in itrs.split(' '):
                filename = f'{folder}/linux.np.server.log.{run_id}_{core}_{msg}_5000_{itr}_{dv}_135.csv'
                df = pd.read_csv(filename, sep = ' ')
                d = df['rx_bytes'].value_counts()

                ## filter out 0's
                d = d[d.index > 0]
                sumt = df['rx_bytes'].sum()
                total = d.sum()
                d = d/total
                
                entropy = -(d * np.log(d)).sum()
                enl.append(entropy)
                #print(f'MSG={msg} ITR={itr} DVFS={dv} RX_BYTE_OVERHEAD={(sumt/(8192*5000)) - 1.0} NUM_INTERRUPTS={df.shape[0]} ENTROPY={entropy}')
                
                ovrhl.append((sumt/(float(msg)*5000)) - 1.0)
                nitr.append(df.shape[0])

                df.reset_index(inplace=True)
                df_non0j = df[df['joules']>0].copy()
                df_non0j['timestamp'] = df_non0j['timestamp'] - df_non0j['timestamp'].min()
                df_non0j['joules'] = df_non0j['joules'] - df_non0j['joules'].min()
    
                df_non0j['timestamp'] = df_non0j['timestamp'] * TIME_CONVERSION_khz
                df_non0j['joules'] = df_non0j['joules'] * JOULE_CONVERSION                
                
                #get edp value
                last_row = df_non0j.tail(1).iloc[0]
                edp_val = 0.5 * last_row['joules'] * last_row['timestamp']
                edpl.append(edp_val)
                tputl.append(int(msg)*5000 / float(last_row['timestamp']))

                lastr = last_row['timestamp']

                # ref cycle diff
                df_non0j['ref_cycles_diff'] = df_non0j['ref_cycles'].diff()
                df_non0j['nonidle_timestamp_diff'] = df_non0j['ref_cycles_diff'] * TIME_CONVERSION_khz
                mmean = df_non0j['nonidle_timestamp_diff'].mean()
                mmedian = df_non0j['nonidle_timestamp_diff'].median()
                df_non0j['index_diff'] = df_non0j['index'].diff()
                imean = df_non0j['index_diff'].mean()
                imedian = df_non0j['index_diff'].median()
                istd = df_non0j['index_diff'].std()
                print(f'MSG={msg} ITR={itr} RX_BYTE_OVERHEAD={(sumt/(float(msg)*5000)) - 1.0} NUM_INTERRUPTS={df.shape[0]} ENTROPY={entropy} TS={lastr} REFCYCLE={mmean, mmedian} index_mean={imean, imedian, istd}')

        
            if plot_default:
                xaxisp = ['1']+itrs.split(' ')
            else:
                xaxisp = itrs.split(' ')

            #plt.figure()
            #plt.title(f'RUN_ID={run_id} | MSGS={msg} | ITRS={itr_min}...{itr_max}')
            #plt.plot(xaxisp, enl, '*')
            #plt.xlabel('ITR-Delay')
            #plt.ylabel('RX_BYTES Entropy')
            #plt.legend()
            #plt.grid()

            plt.figure()
            plt.title(f'RUN_ID={run_id} | MSGS={msg} | ITRS={itr_min}...{itr_max}')
            plt.plot(xaxisp, edpl, '*')
            plt.xlabel('ITR-Delay')
            plt.ylabel('EDP')
            plt.grid()
            
            plt.figure()
            plt.title(f'RUN_ID={run_id} | MSGS={msg} | ITRS={itr_min}...{itr_max}')
            plt.plot(xaxisp, tputl, '*')
            plt.xlabel('ITR-Delay')
            plt.ylabel('Throughput')
            plt.grid()
            
            #plt.figure()
            #plt.title(f'RUN_ID={run_id} | MSGS={msg} | ITRS={itr_min}...{itr_max}')
            #plt.plot(xaxisp, nitr, '*')
            #plt.xlabel('ITR-Delay')
            #plt.ylabel('NUM INTERRUPTS')
            #plt.grid()

            enl=[]
            edpl=[]
            ovrhl=[]
            nitr=[]
            tputl=[]

    
def netpipe_rx_bytes_plots(folder='', run_id=0, msgs='8192', core=1, dvfs='0x1d00', rapl=135, itrs='2', plot_default=False):
    for msg in msgs.split(' '):
        ## plot Linux Default
        if plot_default:
            filename = f'{folder}/linux.np.log.{run_id}_{core}_{msg}_5000_1_0xFFFF_135.csv'
            print(filename)
            df = pd.read_csv(filename, sep = ' ')
            df['timestamp'] -= df['timestamp'].min()
            df['timestamp'] *= TIME_CONVERSION_khz

            plt.figure()
            #plt.plot(df['timestamp'], df['rx_bytes'], 'p')
            plt.plot(df['rx_bytes'], 'p')
            plt.xlabel('interrupt index')
            plt.title(f'MSG={msg} Linux Default rx_bytes')
            
        for d in dvfs.split(' '):
            for itr in itrs.split(' '):
                filename = f'{folder}/linux.np.log.{run_id}_{core}_{msg}_5000_{itr}_{d}_135.csv'
                print(filename)
                df = pd.read_csv(filename, sep = ' ')
                df['timestamp'] -= df['timestamp'].min()
                df['timestamp'] *= TIME_CONVERSION_khz
                
                plt.figure()
                df['timestamp_diff'] = df['timestamp'].diff()
                #plt.plot(df['timestamp_diff'], df['rx_bytes'], 'p')
                plt.plot(df['rx_bytes'], 'p')
                plt.xlabel('interrupt index')
                plt.title(f'RX_BYTES\n MSG={msg} ITR={itr}')
    
def netpipe_edp_plots(folder='', run_id=0, msgs='8192', core=1, dvfs='0x1d00', rapl=135, itrs='2', plot_default=False):
    if folder == '':
        print("Need folder\n")
        exit(0)
        
    x_offset, y_offset = 0.01/5, 0.01/5    
        
    for msg in msgs.split(' '):
        fig = plt.figure(figsize=(9,7))                
        ## plot Linux Default
        if plot_default:
            filename = f'{folder}/linux.np.log.{run_id}_{core}_{msg}_5000_1_0xFFFF_135.csv'
            print(filename)
            df = pd.read_csv(filename, sep = ' ')
            J, T,_ = plot(df, 'Linux Default', projection=True, include_edp_label=False, JOULE_CONVERSION=JOULE_CONVERSION, TIME_CONVERSION=TIME_CONVERSION_khz)
            plt.text(x_offset + T, y_offset + J, f'(-, -, *)')
        
        for d in dvfs.split(' '):
            for itr in itrs.split(' '):
                filename = f'{folder}/linux.np.log.{run_id}_{core}_{msg}_5000_{itr}_{d}_135.csv'
                print(filename)
                df = pd.read_csv(filename, sep = ' ')
                sumt = df['rx_bytes'].sum()
                
                print(filename, 'rxbytes=', sumt/(8192*5000), df.shape[0])                
                J, T,_ = plot(df, 'Linux Tuned', projection=True, include_edp_label=False, JOULE_CONVERSION=JOULE_CONVERSION, TIME_CONVERSION=TIME_CONVERSION_khz)
                                
                plt.text(x_offset + T, y_offset + J, f'({itr}, {d}, *)')
        
        #axes etc.
        #prettify()
        plt.xlabel('Time (seconds)')
        plt.ylabel('Energy Consumed (Joules)')
        
        plt.xlim(0, plt.xlim()[1])
        plt.ylim(0, plt.ylim()[1]) 
        
        plt.grid()
        #plt.title(f"Netpipe Experiment\n Msg Size = {msg} Bytes")
        plt.title(f"{folder}\n Msg Size = {msg} Bytes")
        
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
