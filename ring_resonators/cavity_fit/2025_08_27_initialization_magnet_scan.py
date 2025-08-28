import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_4/10mK/2025_08_27')
FREQ_START = 194810.893  # unit: GHz
FREQ_END = 194819.627

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
ref_freq = 194810
color_pre = 'cornflowerblue'
color_post = 'coral'


# load files
files = glob.glob('cavity_scan_5min_350init_*.txt', root_dir=DATA_DIR)
files = sorted(files)

# get frequency array from first file
df = pd.read_csv(os.path.join(DATA_DIR, files[0]), sep='\t')
time = df['time(ps)']
freq = np.linspace(FREQ_START-ref_freq, FREQ_END-ref_freq, len(time))

for file in files:
    field = file[-9:-6]

    df = pd.read_csv(os.path.join(DATA_DIR, file), sep='\t')
    counts = df['counts']

    plt.plot(freq, counts, label=f'{field} mT')

plt.legend()
plt.xlim((5, 6.25))

plt.tight_layout()
plt.show()
