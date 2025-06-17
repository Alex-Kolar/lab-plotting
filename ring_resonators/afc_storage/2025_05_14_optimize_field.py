import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


OFFRES = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
          "/New_mounted_device/10mK/2025_05_14/afc/afc_storage_experiment.npz")
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/2025_05_14/afc")
FILE_FMT = "afc_storage_experiment_{}.npz"
# map file number to magnetic field (mT)
FIELDS = {8: 100,
          3: 120,
          5: 140,
          6: 160,
          7: 180}

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (1.45, 1.55)
ylim = (0, 60)


# gather data
offres_data = np.load(OFFRES, allow_pickle=True)
burn_field = []
burn_data = []
for file_no, time in FIELDS.items():
    burn_field.append(time)
    file_path = os.path.join(DATA_DIR, FILE_FMT.format(file_no))
    data = np.load(file_path, allow_pickle=True)
    burn_data.append(data)


# plot echos
for field, data in zip(burn_field, burn_data):
    plt.plot(data['bins'], data['counts'],
             label=f'{field} mT')

plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.legend()
plt.xlim(xlim)
plt.ylim(ylim)

plt.tight_layout()
plt.show()
