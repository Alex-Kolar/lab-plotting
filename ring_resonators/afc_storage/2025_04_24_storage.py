import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


OFFRES_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/New_mounted_device/10mK/2025_04_24/afc/testing/afc_storage_experiment_2.npz")
STORAGE_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                "/New_mounted_device/10mK/2025_04_24/afc/testing/afc_storage_experiment.npz")
RESET_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/New_mounted_device/10mK/2025_04_24/afc/testing/afc_storage_experiment_1.npz")



# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_offres = 'cornflowerblue'
color = 'coral'
color_reset = 'mediumpurple'
PLOT_PEAKS = False


# read data
data_offres = np.load(OFFRES_DATA)
data_echo = np.load(STORAGE_DATA)
data_reset = np.load(RESET_DATA)

# plotting
time = data_offres['bins']
plt.plot(time, data_offres['counts'],
         color=color_offres)

plt.title('Off-Resonant Pulse Measurement')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.yscale('log')
plt.xlim(0.9, 1.2)

plt.tight_layout()
plt.show()

time = data_echo['bins']
plt.plot(time, data_echo['counts'],
         color=color)

plt.title('Echo Measurement')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.yscale('log')
plt.xlim(0.9, 1.2)

plt.tight_layout()
plt.show()

time = data_reset['bins']
plt.plot(time, data_reset['counts'],
         color=color_reset)

plt.title('Reset Echo Measurement')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.yscale('log')
plt.xlim(0.9, 1.2)

plt.tight_layout()
plt.show()


# plotting overlaid
time = data_echo['bins']
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
plt.xlim(0.95, 1.2)

plt.tight_layout()
plt.show()


# plotting overlaid
CUTOFF = 1.08
time = data_echo['bins']
time_diff = time[1] - time[0]
idx_to_magnify = np.where(time >= CUTOFF)[0]
plot_offres = data_offres['counts']
plot_offres[idx_to_magnify] = plot_offres[idx_to_magnify] * 100
plot_echo = data_echo['counts']
plot_echo[idx_to_magnify] = plot_echo[idx_to_magnify] * 100
plt.bar(time, plot_offres,
        width=time_diff,
        color=color_offres, alpha=0.5,
        label='Off-Resonant Pulse')
plt.bar(time, plot_echo,
        width=time_diff,
        color=color, alpha=0.5,
        label='Echo Pulse')
plt.axvline(CUTOFF, linestyle='--', color='black')

plt.title('Echo Measurement')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.legend(shadow=True)
# plt.yscale('log')
plt.xlim(0.95, 1.2)

plt.tight_layout()
plt.show()
