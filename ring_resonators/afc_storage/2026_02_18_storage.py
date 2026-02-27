import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


OFFRES_DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
               '/Mounted_device_mk_5/10mK/2026_02_18/afc/afc_storage_experiment_offres.npz')
STORAGE_DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                '/Mounted_device_mk_5/10mK/2026_02_18/afc/afc_storage_experiment_100MHz_500kHz.npz')
edge_coupling_efficiency = 0.07
snspd_efficiency = 0.5

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
plt.legend(framealpha=1)
# plt.yscale('log')
plt.xlim(xlim)

plt.tight_layout()
plt.show()

# calculate efficiencies
idx_in = np.where(np.logical_and(time > 0.9, time < 1.1))[0]
idx_echo = np.where(np.logical_and(time > 1.9, time < 2.1))[0]
counts_in = np.sum(data_offres['counts'][idx_in])
counts_echo = np.sum(data_echo['counts'][idx_echo])
print('Counts In:', counts_in)
print('Counts Echo:', counts_echo)
print('Efficiency:', counts_echo/counts_in)

# calculate pulse statistics
# params = data_offres['storage parameters']
# pulse_freq = data_offres['storage parameters']['pulse_freq']
# integration_time = data_offres['storage parameters']['integration_time']
integration_time = 300
pulse_freq = 100e3
num_pulses = integration_time * pulse_freq
total_photons = counts_in / edge_coupling_efficiency / snspd_efficiency
photons_per_pulse = total_photons / num_pulses
print('Total photons on chip:', total_photons)
print('Photons per pulse:', photons_per_pulse)

