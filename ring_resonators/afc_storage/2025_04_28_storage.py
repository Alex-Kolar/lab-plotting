import os
import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/2025_04_28/afc_sweep")


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


# gather and read data
files = glob.glob('afc_storage_experiment*.npz', root_dir=DATA_DIR)
all_data = []
for file in files:
    full_path = os.path.join(DATA_DIR, file)
    all_data.append(np.load(full_path, allow_pickle=True))


# # do test plotting
# for data in all_data:
#     plt.plot(data['bins'], data['counts'])
#
# plt.xlim(1.9, 2.1)
# plt.ylim(0, 10)
# plt.tight_layout()
# plt.show()


for data in all_data:
    print(data['burn power'])
