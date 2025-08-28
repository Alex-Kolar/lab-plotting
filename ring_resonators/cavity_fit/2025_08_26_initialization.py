import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


DATA_PRE = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_4/10mK/2025_08_26/cavity_scan_5min_preinit.txt')
DATA_POST = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
             '/Mounted_device_mk_4/10mK/2025_08_26/cavity_scan_5min_postinit.txt')
FREQ_START = 194810.893  # unit: GHz
FREQ_END = 194819.627
SWEEP_START = 194814.839 + 0.6  # unit: GHz (and AOM offset)
SWEEP_END = 194815.781 + 0.6

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
ref_freq = 194810
color_pre = 'cornflowerblue'
color_post = 'coral'


# load files
df_pre = pd.read_csv(DATA_PRE, sep='\t')
df_post = pd.read_csv(DATA_POST, sep='\t')

time = df_pre['time(ps)']
freq = np.linspace(FREQ_START-ref_freq, FREQ_END-ref_freq, len(time))


# fig, ax = plt.subplots(figsize=(12, 6))
plt.plot(freq, df_pre['counts'], color=color_pre,
         label='Before Initialization')
plt.plot(freq, df_post['counts'], color=color_post,
         label='After Initialization')
# plt.axvline(SWEEP_START-ref_freq, color='k', ls='--')
# plt.axvline(SWEEP_END-ref_freq, color='k', ls='--')
plt.xlim((3, 8))
plt.xlabel(f'Detuning (GHz) from {ref_freq:.0f} GHz')
plt.ylabel('Counts')
plt.legend()

plt.tight_layout()
plt.show()
