import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import GaussianModel, ConstantModel

DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/2025_06_17/afc")
FILE_FMT = "afc_storage_experiment_{}.npz"
# map file number to atten, burn time
TIMES = {
    # 1: (30, 20),
    # 2: (20, 5),
    # 3: (20, 10),
    4: (10, 5),
    5: (10, 10)
}
cav_contrast = 0.7830598775493341

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (1.4, 1.6)  # for pulse
xlim_input = (0.9, 1.1)  # for input
xlim_full = (0.9, 1.6)  # for both
ylim = (0, 20)
PLOT_ALL_STORAGE = True

# gather data
burn_times = []
burn_attens = []
burn_data = []
for file_no, (atten, time) in TIMES.items():
    burn_times.append(time)
    burn_attens.append(atten)
    file_path = os.path.join(DATA_DIR, FILE_FMT.format(file_no))
    data = np.load(file_path, allow_pickle=True)
    burn_data.append(data)

    # get input counts
    count_times = data['bins']
    counts = data['counts']
    idx_to_sum = np.where(np.logical_and(count_times > xlim_input[0],
                                         count_times <= xlim_input[-1]))[0]
    total_input = np.sum(counts[idx_to_sum])

    # get output counts
    idx_to_sum = np.where(np.logical_and(count_times > xlim[0],
                                         count_times <= xlim[-1]))[0]
    total_output = np.sum(counts[idx_to_sum])

    # get efficiency (including cavity)
    efficiency = total_output / (total_input / (1 - cav_contrast))
    print(efficiency)

# plot echos
for atten, time, data in zip(burn_attens, burn_times, burn_data):
    plt.plot(data['bins'], data['counts'],
             label=f'{time} Minute Burn, {atten} dB attenuation')

plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.legend()
plt.xlim(xlim)
plt.ylim(ylim)
# plt.yscale('log')

plt.tight_layout()
plt.show()


# plot echo and input
for atten, time, data in zip(burn_attens, burn_times, burn_data):
    plt.plot(data['bins'], data['counts'],
             label=f'{time} Minute Burn, {atten} dB attenuation')

plt.xlabel(r'Time ($\mathrm{\mu}$s)')
plt.ylabel('Counts')
plt.legend()
plt.xlim(xlim_full)
plt.yscale('log')

plt.tight_layout()
plt.show()
