import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


STORAGE_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                "/New_mounted_device/10mK/2025_06_06/afc/afc_storage_experiment.npz")


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color_offres = 'cornflowerblue'
color = 'coral'
PLOT_PEAKS = False
xlim = (0.8, 1.7)


# read data
data_echo = np.load(STORAGE_DATA, allow_pickle=True)


# plotting overlaid
time = data_echo['bins']
time_diff = time[1] - time[0]
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

