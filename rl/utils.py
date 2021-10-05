import os
import glob
import numpy as np

def get_params(loc, tag='linux.mcd.dmesg'):
    fnames = glob.glob(f'{loc}/*')

    tag_dict = {'core': [],
                'run': [],
                'itr': [],
                'dvfs': [],
                'rapl': [],
                'qps': []}

    fname_dict = {}

    for f in fnames:
        fields = f.split('/')[-1].split('.')[-1].split('_')
        if len(fields)==6: #excluding default
            run, core, itr, dvfs, rapl, qps = fields

            tag_dict['run'].append(run)
            tag_dict['core'].append(core)
            tag_dict['itr'].append(itr)
            tag_dict['dvfs'].append(dvfs)
            tag_dict['rapl'].append(rapl)
            tag_dict['qps'].append(qps)

            fname_dict[f.split('/')[-1]] = 1

    return tag_dict, fname_dict

def check_grid(tag_dict, fname_dict):
    uniq_itr = np.unique(tag_dict['itr'])
    uniq_dvfs = np.unique(tag_dict['dvfs'])
    uniq_rapl = np.unique(tag_dict['rapl'])
    uniq_qps = np.unique(tag_dict['qps'])
    uniq_cores = np.unique(tag_dict['core'])

    run = 0
    missing_list = []
    for itr in uniq_itr:
        for dvfs in uniq_dvfs:
            for rapl in uniq_rapl:
                for qps in uniq_qps:
                    for core in uniq_cores:
                        fname = f'linux.mcd.dmesg.{run}_{core}_{itr}_{dvfs}_{rapl}_{qps}'
                        if fname not in fname_dict:
                            print(fname)
                            missing_list.append(fname)

    return missing_list


