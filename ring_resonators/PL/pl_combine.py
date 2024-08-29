import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel


DATA_DIRS = [("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/New_mounted_device/10mK/pl_08282024"),
             ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/New_mounted_device/10mK/pl_08292024")]
CUTOFF_IDX = 5
NORMALIZE = True  # normalize to first time bin (laser pulse)

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')

pl_files = []
for dir in DATA_DIRS:
    pl_files += glob.glob(dir + "/*.npz")

model = ExponentialModel() + ConstantModel()
freqs = []
all_res = []
laser_pulses = []
for file in pl_files:
    freq_str = os.path.basename(file).split('.')[0]
    freq_str_decimal = freq_str.replace('_', '.')
    freq = float(freq_str_decimal)
    freqs.append(freq)

    data = np.load(file)
    bins = data['bins'][CUTOFF_IDX:]
    hist = data['hist'][CUTOFF_IDX:]
    laser_pulses.append(data['hist'][0])

    res = model.fit(hist, x=bins)
    all_res.append(res)

# define frequency array
freqs = np.array(freqs)
freq_min = min(freqs)
freqs = freqs - freq_min

# get data from fits
amplitudes = np.array(list(map(lambda x: x.params['amplitude'].value, all_res)))
amplitude_err = np.array(list(map(lambda x: x.params['amplitude'].stderr, all_res)))
bgs = np.array(list(map(lambda x: x.params['c'].value, all_res)))
bg_err = np.array(list(map(lambda x: x.params['c'].stderr, all_res)))

# normalize amplitudes
laser_pulses = np.array(laser_pulses)
max_pulse = np.max(laser_pulses)
for i, pulse in enumerate(laser_pulses):
    scaling_factor = max_pulse / pulse
    amplitudes[i] *= scaling_factor
    amplitude_err[i] *= scaling_factor


# plotting of PL
plt.errorbar(freqs, amplitudes, yerr=amplitude_err,
             ls='', marker='o', capsize=3, color='cornflowerblue')
plt.xlabel(f'Frequency + {freq_min} (GHz)')
plt.ylabel('Fitted PL Amplitude (A.U.)')
plt.grid(True)

plt.tight_layout()
plt.show()


# plotting of initial pulse size
fig, ax = plt.subplots()
ax2 = ax.twinx()

ax.errorbar(freqs, bgs, yerr=bg_err,
            ls='', marker='o', capsize=3, color='cornflowerblue')
ax2.plot(freqs, laser_pulses,
         ls='', marker='o', color='coral')
ax.set_xlabel(f'Frequency + {freq_min} (GHz)')
ax.set_ylabel('Fitted Background (A.U.)',
              color='cornflowerblue')
ax2.set_ylabel(f'Laser Pulse Counts',
               color='coral')
ax.grid(True)

plt.tight_layout()
plt.show()
