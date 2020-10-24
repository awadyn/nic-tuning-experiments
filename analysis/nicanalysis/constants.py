JOULE_CONVERSION = 0.00001526 #counter * constant -> Joules
ALT_TIME_CONVERSION_khz = 1./(2199999*1000) # neu-15-3: Intel(R) Xeon(R) CPU E5-2660 0 @ 2.20GHz
TIME_CONVERSION_khz = 1./(2899999*1000)

TPUT_CONVERSION_FACTOR = 2 * 8 / (1024*1024)

COLS = ['i', 'rx_desc', 'rx_bytes', 'tx_desc', 'tx_bytes', 'instructions', 'cycles', 'llc_miss', 'joules', 'timestamp']
COLORS = {'linux_default': 'blue',
          'linux_tuned': 'green',
          'ebbrt_tuned': 'red'}          
LABELS = {'linux_default': 'Linux Default',
          'linux_tuned': 'Linux Tuned',
          'ebbrt_tuned': 'Library OS Tuned'}
COLORMAPS = {'linux_tuned': 'Greens',
             'ebbrt_tuned': 'Reds'}
