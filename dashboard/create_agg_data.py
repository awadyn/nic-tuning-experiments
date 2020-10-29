from config import *
from process_log_data import *
import pandas as pd
import glob

def create_agg_file(loc, searchstring, pass_colnames, skiprows):
    data = {}    
    for filename in glob.glob(f'{loc}/{searchstring}'):
        df, df_orig = process_log_file(filename, pass_colnames=pass_colnames, skiprows=skiprows)

        r = {}
        r['sys'] = 
        r['i'] = 
        r['itr'] = 
        r['dvfs'] = 
        r['rapl'] = 
        r['']