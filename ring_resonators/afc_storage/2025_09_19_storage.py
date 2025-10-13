import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


OFFRES_DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
               '/Mounted_device_mk_4/10mK/2025_09_19/afc/AFC_offres_1min.txt')
STORAGE_DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                '/Mounted_device_mk_4/10mK/2025_09_19/afc/AFC_storage_10min.txt')


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_offres = 'cornflowerblue'
color = 'coral'
PLOT_PEAKS = False
xlim = (0.95, 1.2)


# read data
# data_offres = np.load(OFFRES_DATA, allow_pickle=True)
# data_echo = np.load(STORAGE_DATA, allow_pickle=True)
data_offres = pd.read_csv(OFFRES_DATA, sep='\t')
data_echo = pd.read_csv(STORAGE_DATA, sep='\t')


# plotting overlaid
time = data_offres['Time(ps)'] / 1e6  # convert to us
time *= -1  # trigger is backwards
time_diff = time[1] - time[0]
plt.bar(time, data_offres['Counts'] * 10,  # account for difference in integration time (1 vs. 10 minutes)
        width=time_diff,
        color=color_offres, alpha=0.5,
        label=r'Off-Resonant Pulse ($\times$10)')
plt.bar(time, data_echo['Counts'],
        width=time_diff,
        color=color, alpha=0.5,
        label='Echo Pulse')

plt.title('Echo Measurement')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.legend(shadow=True)
plt.yscale('log')
plt.xlim(xlim)

plt.tight_layout()
plt.show()

# calculate efficiencies
idx_in = np.where(np.logical_and(time > 0.97, time < 1.07))[0]
idx_echo = np.where(np.logical_and(time > 1.07, time < 1.17))[0]
counts_in = np.sum(data_offres['Counts'][idx_in]) * 10
counts_echo = np.sum(data_echo['Counts'][idx_echo])
print('Efficiency:', counts_echo/counts_in)


# # plotting overlaid (better)
# idx_to_plot = np.where(np.logical_and(time >= xlim[0], time <= xlim[1]))[0]
# time_to_plot = time[idx_to_plot]
# offres_to_plot = data_offres['counts'][idx_to_plot]
# echo_to_plot = data_echo['counts'][idx_to_plot]
#
# fig, ax = plt.subplots(figsize=(4.5, 4), dpi=400)
#
# ax.bar(time_to_plot, offres_to_plot,
#        width=time_diff,
#        color=color_offres, alpha=0.5,
#        label='Off-Resonant Pulse')
# ax.bar(time_to_plot, echo_to_plot,
#        width=time_diff,
#        color=color, alpha=0.5,
#        label='Echo Pulse')
#
# ax.set_title('Echo Measurement')
# ax.set_xlabel(r'Time ($\mathrm{\mu}$s)')
# ax.set_ylabel('Counts')
# ax.legend(frameon=False)
# ax.set_yscale('log')
# ax.set_xlim(xlim)
#
# fig.tight_layout()
# fig.show()
