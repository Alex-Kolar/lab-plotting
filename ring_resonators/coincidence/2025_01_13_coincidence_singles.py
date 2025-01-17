import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# data params
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/300K_no_erbium/coincidence/01132025")
DATA_LIST = [
    ("Dark Counts", "dark_counts"),
    ("Ambient Lighting", "lights_on"),
    ("Raman Scattering", "laser_off_res"),
    ("Generated Photons", "laser_on_res")
]

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim_range = 15  # size of x lims
color_signal = 'cornflowerblue'
color_idler = 'coral'


counts_avg = [[] for _ in range(len(DATA_LIST))]
counts_std = [[] for _ in range(len(DATA_LIST))]
for i, (label, filename) in enumerate(DATA_LIST):
    df = pd.read_csv(DATA_DIR + "/" + filename + ".txt", sep='\t')

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
x_ticks = np.arange(len(DATA_LIST))
x_width = 0.4

signal_avg = [data[0] for data in counts_avg]
signal_std = [data[0] for data in counts_std]
idler_avg = [data[1] for data in counts_avg]
idler_std = [data[1] for data in counts_std]
labels = [data[0] for data in DATA_LIST]

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
plt.yscale('log')

plt.tight_layout()
plt.show()
