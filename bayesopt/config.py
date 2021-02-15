class Locations:
    aggregate_files_loc = 'summary_data'

    netpipe_logs_loc = '/home/sanjay/sesa-papers/ukpwr/logs/aug19_netpipelogs/rapl135'
    netpipe_linux_subfolder = 'governor'
    netpipe_ebbrt_subfolder = 'rapl135'

    nodejs_logs_loc = '/home/sanjay/sesa-papers/ukpwr/logs/aug14_nodejslogs'
    nodejs_linux_subfolder = ''
    nodejs_ebbrt_subfolder = ''

    mcd_logs_loc = '/home/sanjay/sesa-papers/ukpwr/logs/aug19_mcdlogs'
    mcd_linux_subfolder = ''
    mcd_ebbrt_subfolder = ''

    mcdsilo_logs_loc = '/home/sanjay/sesa-papers/ukpwr/logs/aug19_mcdsilologs'
    mcdsilo_linux_subfolder = ''
    mcdsilo_ebbrt_subfolder = ''

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


msg_qps_sizes = {
  'netpipe': {'options': [64, 8192, 65536, 524288], 'value': 8192},
  'nodejs': {'options': [], 'value': None},
  'mcd': {'options': [200000, 400000, 600000], 'value': 200000},
  'mcdsilo': {'options': [50000, 100000, 200000], 'value': 50000},
}


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
hover_data = {
  'netpipe': ['itr', 'rapl', 'dvfs', 'Sys', 'instructions_mean', 'num_interrupts_mean'],
  'nodejs': ['itr', 'rapl', 'dvfs', 'Sys', 'instructions_mean'],
  'mcd': ['itr', 'rapl', 'dvfs', 'Sys', 'instructions_mean'],
  'mcdsilo': ['itr', 'rapl', 'dvfs', 'Sys', 'instructions_mean']
}