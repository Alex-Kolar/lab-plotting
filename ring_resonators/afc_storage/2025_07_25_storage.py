import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


OFFRES_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/Mounted_device_mk_3/10mK/2025_07_25/afc/afc_5minoffres.npz")
STORAGE_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                "/Mounted_device_mk_3/10mK/2025_07_25/afc/afc_5min20dbburn_5minstorage_10tooth_10MHzspacing.npz")


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_offres = 'cornflowerblue'
color = 'coral'
PLOT_PEAKS = False
xlim = (0.8, 1.7)


# read data
data_offres = np.load(OFFRES_DATA, allow_pickle=True)
data_echo = np.load(STORAGE_DATA, allow_pickle=True)


# plotting overlaid
time = data_offres['bins']
time_diff = time[1] - time[0]
plt.bar(time, data_offres['counts'],
        width=time_diff,
        color=color_offres, alpha=0.5,
        label='Off-Resonant Pulse')
plt.bar(time, data_echo['counts'],
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
