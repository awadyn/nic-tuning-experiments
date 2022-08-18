import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import os, sys, shutil, re

plt.ion()

def process_logs(PATH):
	data = []
	for f in os.listdir(PATH):
		#extract from filename
		itr = re.search(r'(.*?)\.(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv', f).group(4)
		dvfs = re.search(r'(.*?)\.(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)\.csv', f).group(5)

		features = process_file(PATH + '/' + f, itr, dvfs)
		data.append(features)

	data = pd.DataFrame(data)
	print("Data: ")
	print(data)
	print()
	print()

	return data

def process_file(path, itr, dvfs):
	features = {}
	values = pd.read_csv(path, sep = ' ', skiprows = 1, index_col = 0, names = ['rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'ref_cycles', 'llc_miss', 'c3', 'c6', 'c7', 'joules', 'timestamp'])

	PCT_LIST = [10, 25, 50, 75, 90, 99]
	
	interrupt_cols = ['rx_bytes', 'rx_desc', 'tx_bytes', 'tx_desc']
	for c in interrupt_cols:
		pcts = np.percentile(values[c], PCT_LIST)
		for i, p in enumerate(PCT_LIST):
			features[f'{c}_{p}'] = pcts[i]
	    
	ms_cols = ['instructions', 'llc_miss', 'cycles', 'ref_cycles']
	for c in ms_cols:
		pcts = np.percentile(values[c], PCT_LIST)
		for i, p in enumerate(PCT_LIST):
			features[f'{c}_{p}'] = pcts[i]

	perf_cols = ['joules']
	for c in perf_cols:
		pct = np.percentile(values[c], 99)
		features[f'{c}_99'] = pct
		print("joules: ", pct)

	#extract from filename
	features['itr'] = itr
	features['dvfs'] = dvfs
#	
#	features['lat_99'] = ...
	return features

if __name__ == '__main__':
	path = sys.argv[1]
	print(path)
	data = process_logs(path)
	data.to_csv("./features/logs_0_percentiles.csv", sep = ' ')

