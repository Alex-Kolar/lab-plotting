import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# data params
DATA_ONRES = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/New_mounted_device/4K/06032025/Counter_2025-06-03_17-48-43_onres.txt")
DATA_OFFRES = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/New_mounted_device/4K/06032025/Counter_2025-06-03_17-49-45_offres.txt")

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_signal = 'cornflowerblue'
color_idler = 'coral'


counts_avg = [[], []]
counts_std = [[], []]
for i, path in enumerate((DATA_ONRES, DATA_OFFRES)):
    df = pd.read_csv(path, sep='\t')

    # get signal data
    signal_avg_counts = np.mean(df['counts(1/s)'])
    signal_std_counts = np.std(df['counts(1/s)'])
    counts_avg[i].append(signal_avg_counts)
    counts_std[i].append(signal_std_counts)

    # get idler data
    idler_avg_counts = np.mean(df['counts(1/s).1'])
    idler_std_counts = np.std(df['counts(1/s).1'])
    counts_avg[i].append(idler_avg_counts)
    counts_std[i].append(idler_std_counts)


# plotting
x_ticks = np.arange(2)
x_width = 0.4

signal_avg = [data[0] for data in counts_avg]
signal_std = [data[0] for data in counts_std]
idler_avg = [data[1] for data in counts_avg]
idler_std = [data[1] for data in counts_std]
labels = ["On-Resonant", "Off-Resonant"]

plt.bar(x_ticks-(x_width/2), signal_avg,
        label='Signal Channel',
        width=x_width, color=color_signal, edgecolor='k', zorder=3)
plt.errorbar(x_ticks-(x_width/2), signal_avg, yerr=signal_std,
             marker='', ls='', color='k', capsize=3, zorder=4)
plt.bar(x_ticks+(x_width/2), idler_avg,
        label='Idler Channel',
        width=x_width, color=color_idler, edgecolor='k', zorder=3)
plt.errorbar(x_ticks+(x_width/2), idler_avg, yerr=idler_std,
             marker='', ls='', color='k', capsize=3, zorder=4)

plt.ylabel(r"Counts (/s)")
plt.grid(True, axis='y')
plt.xticks(x_ticks, labels, rotation=45)
plt.legend(shadow=True)

plt.tight_layout()
plt.show()
