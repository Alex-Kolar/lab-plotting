import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


OFFRES_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/Bulk_crystal/10mK/02212025/Echo/Off_res_10kHz_10min_38_9db.txt")
STORAGE_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                "/Bulk_crystal/10mK/02212025/Echo/YSO_echo3_10kHz_10min_194812_790GHz_38_9db_lightsoff.txt")


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 10})
color_offres = 'cornflowerblue'
color = 'coral'
PLOT_PEAKS = False


# read data
df_offres = pd.read_csv(OFFRES_DATA, sep='\t')
df_echo = pd.read_csv(STORAGE_DATA, sep='\t')


# find peaks
offres_peak_data = df_offres['counts']
echo_peak_data = df_echo['counts']
offres_idx = find_peaks(offres_peak_data, prominence=10)[0]
echo_idx = find_peaks(echo_peak_data, prominence=10)[0]


# plotting
time = df_offres['time(ps)'] / 1e6  # convert to us
plt.plot(time, df_offres['counts'],
         color=color_offres)

plt.title('Off-Resonant Pulse Measurement')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.yscale('log')
plt.xlim(0, 2.2)

plt.tight_layout()
plt.show()

time = df_echo['time(ps)'] / 1e6  # convert to us
plt.plot(time, df_echo['counts'],
         color=color)

plt.title('Echo Measurement')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.yscale('log')
plt.xlim(0, 2.2)

plt.tight_layout()
plt.show()


# plotting peaks
if PLOT_PEAKS:
    time = df_offres['time(ps)'] / 1e6  # convert to us
    plt.plot(time, df_offres['counts'],
             color=color_offres)
    plt.plot(time[offres_idx], df_offres['counts'][offres_idx],
             color='k', marker='x', ls='')
    for peak_idx in offres_idx:
        peak = df_offres['counts'][peak_idx]
        plt.text(time[peak_idx], peak, f'{peak:.2f}')

    plt.title('Off-Resonant Pulse Measurement')
    plt.xlabel(r'Time ($\mathrm{\mu}$s)')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.xlim(0, 2.2)

    plt.tight_layout()
    plt.show()

    time = df_echo['time(ps)'] / 1e6  # convert to us
    plt.plot(time, df_echo['counts'],
             color=color)
    plt.plot(time[echo_idx], df_echo['counts'][echo_idx],
             color='k', marker='x', ls='')
    for peak_idx in echo_idx:
        peak = df_echo['counts'][peak_idx]
        plt.text(time[peak_idx], peak, f'{peak:.2f}')

    plt.title('Echo Measurement')
    plt.xlabel(r'Time ($\mathrm{\mu}$s)')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.xlim(0, 2.2)

    plt.tight_layout()
    plt.show()


# plotting overlaid
time = df_offres['time(ps)'] / 1e6  # convert to us
time_diff = time[1] - time[0]

fig, ax = plt.subplots(figsize=(4, 3), dpi=400)
ax.bar(time, df_offres['counts'],
        width=time_diff,
        color=color_offres, alpha=0.5,
        label='Off-Resonant Pulse')
ax.bar(time, df_echo['counts'],
        width=time_diff,
        color=color, alpha=0.5,
        label='Echo Pulse')

ax.set_title('Echo Measurement')
ax.set_xlabel(r'Time ($\mathrm{\mu}$s)')
ax.set_ylabel('Counts')
ax.legend(shadow=True)
ax.set_yscale('log')
ax.set_xlim(0, 1.2)

fig.tight_layout()
fig.show()
