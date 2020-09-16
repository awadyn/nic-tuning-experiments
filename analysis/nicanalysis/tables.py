import pandas as pd

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
