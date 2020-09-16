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