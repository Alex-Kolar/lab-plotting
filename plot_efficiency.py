import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# matplotlib parameters
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})

df_counts = pd.read_csv('snspd_bias.csv')
bias = df_counts['Bias (uA)']
counts = df_counts['Countrate 2 (counts/s)']

df_dark = pd.read_csv('Ch9 Dark Count Troubleshooting - Sheet1.csv')
bias_dark = df_dark['Bias current (uA)']
counts_dark = df_dark['Dark count rate (cps)']

# truncate dark counts
bias_min = min(bias)
bias_idx = bias_dark.to_list().index(bias_min)
bias_dark = bias_dark[bias_idx:]
counts_dark = counts_dark[bias_idx:]

# truncate counts
bias_max = max(bias_dark)
bias_idx = bias.to_list().index(bias_max)
bias = bias[:bias_idx]
counts = counts[:bias_idx]

# get dark counts interpolated at measured points
counts_dark_interp = np.interp(bias, bias_dark, counts_dark)

fig, ax = plt.subplots()
ax.plot(bias, counts, '-o', label='Total count rate')
ax.plot(bias_dark, counts_dark, '-o', label='Dark count rate')
ax.plot(bias, counts - counts_dark_interp, '-o', label='Total - dark count rate')

ax.grid('on')

ax.set_xlabel(r'Bias Current ($\mu$A)')
ax.set_ylabel(r'Counts per Second')
ax.legend(shadow=True)

plt.tight_layout()
plt.savefig('count_no_dark')
plt.show()
