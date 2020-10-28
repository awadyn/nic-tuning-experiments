
class Locations:
    aggregate_files_loc = 'summary_data'

    netpipe_logs_loc = '/home/sanjay/sesa-papers/ukpwr/logs/jul20'
    netpipe_logs_governor_folder = 'governor'
    netpipe_logs_tuned_folder = 'rapl135'

    nodejs_logs_loc = '/home/sanjay/sesa-papers/ukpwr/logs/aug13_nodejslogs'
'''
    nodejs_logs_loc = 
    mcd_logs_loc = 
    mcdsilo_logs_loc = 
'''

class PlotList:
    timeline_plots_metrics = ['tx_bytes', 'rx_bytes', 'joules_diff', 'timestamp_diff']

JOULE_CONVERSION = 0.00001526 #counter * constant -> Joules
TIME_CONVERSION_khz = 1./(2899999*1000)
COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'llc_miss', 'joules', 'timestamp']
COLORS = {'linux_default': 'blue',
          'linux_tuned': 'green',
          'ebbrt_tuned': 'red'}          
LABELS = {'linux_default': 'Linux Default',
          'linux_tuned': 'Linux Tuned',
          'ebbrt_tuned': 'Library OS Tuned'}
COLORMAPS = {'linux_tuned': 'Greens',
             'ebbrt_tuned': 'Reds'}


netpipe_msg_sizes = [64, 8192, 65536, 524288]
mcd_qps_sizes = [200000, 400000, 600000]


#nodejs -> netpipe colnames
COL_MAPPER = {'ins': 'instructions',
              'JOULE': 'joules',
              'TSC': 'timestamp',
              'rxdesc': 'rx_desc',
              'rxbytes': 'rx_bytes',
              'txdesc': 'tx_desc',
              'txbytes': 'tx_bytes',
              'cyc': 'cycles',
              'refcyc': 'ref_cycles',
              'llcm': 'llc_miss',
            }