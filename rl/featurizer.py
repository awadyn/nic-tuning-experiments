def compute_features_from_logs(loc, N_parallel=4):
    def compute_entropy(df, colname, nonzero=False):
        x  = df[colname].value_counts()

        if nonzero:
            x = x[x.index > 0]

        x = x / x.sum() #scipy.stats.entropy actually automatically normalizes

        return entropy(x)

    def featurize(file, data):
        '''Modify with new features
        '''
        df = pd.read_csv(file, sep=' ')
        df_non0j = df[df['joules'] > 0].copy()

        #non-zero joules
        df_non0j['timestamp'] = df_non0j['timestamp'] - df_non0j['timestamp'].min()
        df_non0j['joules'] = df_non0j['joules'] - df_non0j['joules'].min()
        
        df_non0j['timestamp'] = df_non0j['timestamp'] * TIME_CONVERSION
        df_non0j['joules'] = df_non0j['joules'] * JOULE_CONVERSION

        #add features here
        d = {'name': file}        
        for col in ['rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes']:
            d[f'entropy_{col}'] = compute_entropy(df, col, nonzero=False)
            d[f'entropy_nonzero_{col}'] = compute_entropy(df, col, nonzero=True)

        tags = file.split('.')[-2]
        rnd, cpu, msg, nrounds, itr, dvfs, rapl = tags.split('_')
        d['rnd'] = int(rnd)
        d['cpu'] = int(cpu)
        d['msg'] = int(msg)
        d['nrounds'] = int(nrounds)
        d['itr'] = int(itr)
        d['dvfs'] = dvfs
        d['rapl'] = int(rapl)

        last_row = df_non0j.tail(1).iloc[0]
        d['joules'] = last_row['joules']
        d['time'] = last_row['timestamp']
        d['edp'] = 0.5 * d['joules'] * d['time']
        d['n_interrupts'] = df.shape[0]
        d['n_nonzero_interrupts'] = df[df['rx_bytes']>0].shape[0]

        cols = ['rx_desc', 'tx_desc', 'rx_bytes', 'tx_bytes']
        corr = df[cols].corr()
        for i in range(len(cols)):
            for j in range(i):
                d[f'corr_{cols[i]}_{cols[j]}'] = corr[cols[i]][cols[j]]

        data.append(d)

    manager = Manager()
    data = manager.list()
    plist = []

    N_current = 0
    for file in glob.glob(f'{loc}'):

        p = Process(target=featurize, args=(file, data))
        p.start()
        N_current += 1

        plist.append(p)

        if N_current % N_parallel == 0:
            [p.join() for p in plist]

    data = list(data)
    data = pd.DataFrame(data)
    data.set_index(['name', 'rnd', 'cpu', 'msg', 'nrounds', 'itr', 'dvfs', 'rapl'], inplace=True)
    
    return data
