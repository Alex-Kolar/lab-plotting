import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


OFFRES_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/New_mounted_device/10mK/PL_holeburn_2025_04_03/storage"
               "/storage off resonant 5 min int 70 ns width with 100kHz rep.txt")
STORAGE_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                "/New_mounted_device/10mK/PL_holeburn_2025_04_03/storage"
                "/Storage low power comb 5 min 70ns width 100kHz rep.txt")


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
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
plt.xlim(3, 3.3)

plt.tight_layout()
plt.show()

time = df_echo['time(ps)'] / 1e6  # convert to us
plt.plot(time, df_echo['counts'],
         color=color)

plt.title('Echo Measurement')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.yscale('log')
plt.xlim(3, 3.3)

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
    plt.xlim(3, 3.3)

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
    plt.xlim(3, 3.3)

    plt.tight_layout()
    plt.show()


# plotting overlaid
time = df_offres['time(ps)'] / 1e6  # convert to us
time_diff = time[1] - time[0]
plt.bar(time, df_offres['counts'],
        width=time_diff,
        color=color_offres, alpha=0.5,
        label='Off-Resonant Pulse')
plt.bar(time, df_echo['counts'],
        width=time_diff,
        color=color, alpha=0.5,
        label='Echo Pulse')

plt.title('Echo Measurement')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.legend(shadow=True)
plt.yscale('log')
plt.xlim(3, 3.3)

plt.tight_layout()
plt.show()
