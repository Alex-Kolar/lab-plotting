import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/2025_04_30/afc_sweep")
SORT_KEY = 'burn power'


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


# gather and read data
files = glob.glob('afc_storage_experiment*.npz', root_dir=DATA_DIR)
all_data = []
for file in files:
    full_path = os.path.join(DATA_DIR, file)
    all_data.append(np.load(full_path, allow_pickle=True))
# sort
key = lambda x: x[SORT_KEY]
all_data = sorted(all_data, key=key)

# do plotting of peaks
for data in all_data:
    atten = data['burn power']
    plt.plot(data['bins'], data['counts'],
             label=f'{atten} dB attenuation')

plt.title('Burn Power Sweep')
plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
# plt.legend()
# plt.xlim(0.8, 1.3)
# plt.ylim(0, 10)
plt.yscale('log')

plt.tight_layout()
plt.show()

# # do plotting of peaks individually
# for data in all_data:
#     atten = data['burn power']
#     plt.plot(data['bins'], data['counts'])
#
#     plt.title(f'{atten} dB Burning')
#     plt.xlabel(r'Time ($\mathrm{\mu}$s)')
#     plt.ylabel('Counts')
#     plt.xlim(0.8, 1.3)
#     plt.ylim(0, 10)
#     # plt.yscale('log')
#
#     plt.tight_layout()
#     plt.show()
