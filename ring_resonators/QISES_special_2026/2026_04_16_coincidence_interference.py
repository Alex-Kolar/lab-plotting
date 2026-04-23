import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model


DATA_CONSTRUCTIVE = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                     '/Unmounted_device_mk_3/2026_04_15/pair_generation/interference_350mV_negP_5min.txt')
DATA_DESTRUCTIVE = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                    '/Unmounted_device_mk_3/2026_04_15/pair_generation/interference_210mV_5min.txt')
num_bins = 10

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


df_constructive = pd.read_csv(DATA_CONSTRUCTIVE, sep='\t')
df_destructive = pd.read_csv(DATA_DESTRUCTIVE, sep='\t')

time = df_constructive['Time(ps)']
time *= 1e-3  # convert to ns

center_bin = np.argmax(df_constructive['Counts'])
idx_start = center_bin - num_bins // 2
idx_end = center_bin + num_bins // 2 + 1

counts_constructive = np.sum(df_constructive['Counts'][idx_start:idx_end])
counts_destructive = np.sum(df_destructive['Counts'][idx_start:idx_end])
visibility = (counts_constructive - counts_destructive) / (counts_constructive + counts_destructive)
print(f'Visibility: {visibility*100:.2f}%')


fig, ax = plt.subplots(figsize=(5, 4), dpi=400)
plt.plot(time, df_constructive['Counts'],
         color='cornflowerblue', label='Constructive')
plt.plot(time, df_destructive['Counts'],
         color='coral', label='Destructive')
plt.title('Franson Interference')
plt.xlabel('Timing Offset (ns)')
plt.ylabel('Coincidence Counts')
plt.legend()
plt.xlim((0, 50))
plt.tight_layout()
plt.show()
