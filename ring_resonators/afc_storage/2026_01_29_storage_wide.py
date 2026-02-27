"""For wide comb (200 MHz) storage.

3 echoes were measured, at the center of the comb and separated by 75 MHz up and down.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


OFFRES_DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
               '/Mounted_device_mk_5/10mK/2026_01_28/afc/off_res.npz')
STORAGE_DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                '/Mounted_device_mk_5/10mK/2026_01_29/afc/40dB_5min_194829040_200MHz.npz')
STORAGE_LOW = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
               '/Mounted_device_mk_5/10mK/2026_01_29/afc/40dB_5min_194828965_200MHz.npz')
STORAGE_HIGH = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                '/Mounted_device_mk_5/10mK/2026_01_29/afc/40dB_5min_194829115_200MHz.npz')


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_offres = 'cornflowerblue'
color = 'coral'
PLOT_PEAKS = False
xlim = (0.9, 2.1)


# read data
data_offres = np.load(OFFRES_DATA, allow_pickle=True)
data_echo = np.load(STORAGE_DATA, allow_pickle=True)
data_echo_low = np.load(STORAGE_LOW, allow_pickle=True)
data_echo_high = np.load(STORAGE_HIGH, allow_pickle=True)


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

# calculate efficiencies
idx_in = np.where(np.logical_and(time > 0.9, time < 1.1))[0]
idx_echo = np.where(np.logical_and(time > 1.9, time < 2.1))[0]
counts_in = np.sum(data_offres['counts'][idx_in])
counts_echo = np.sum(data_echo['counts'][idx_echo])
print('Efficiency:', counts_echo/counts_in)
