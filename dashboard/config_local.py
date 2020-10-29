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