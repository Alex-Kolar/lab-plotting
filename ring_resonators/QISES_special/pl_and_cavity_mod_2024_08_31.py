import glob
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel


# data
PL_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
          "/New_mounted_device/10mK/pl_08312024")
CAVITY_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
              "/New_mounted_device/10mK/09032024/SDS00002.csv")
CAVITY_CALIB_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                    "/New_mounted_device/10mK/09032024/SDS00003.csv")
FREQ_START = 194811.486  # unit: GHz
FREQ_END = 194819.973  # unit: GHz
AOM_FREQ = 0.6  # unit: GHz

# meta idx params
CAVITY_PEAK_RANGE = (3500, 4000)
CUTOFF_IDX = 5

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 14})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')
SAVE_FIG = False
SAVE_NAME = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs"
             "/ring_resonators/new_mounted/10mK_pl/300mT_area_pl_and_cavity.pdf")


# get cavity calibration data
df = pd.read_csv(CAVITY_CALIB_DIR, header=10, skiprows=[11])
ramp = df['CH1'].astype(float).to_numpy()
transmission = df['CH2'].astype(float).to_numpy()

# find difference in scan edges
scan_max = np.argmax(ramp)
trans_max = np.argmax(transmission[CAVITY_PEAK_RANGE[0]:CAVITY_PEAK_RANGE[1]])
idx_diff = (trans_max + CAVITY_PEAK_RANGE[0]) - scan_max

print(f"IDX difference: {idx_diff}")

# get cavity data
df = pd.read_csv(CAVITY_DIR, header=10, skiprows=[11])
ramp = df['CH1'].astype(float).to_numpy()
transmission = df['CH2'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission = transmission[id_min+idx_diff:id_max+idx_diff]

# convert time to frequency
freq_cavity = np.linspace(0, (FREQ_END - FREQ_START), id_max-id_min)  # unit: GHz


# get PL data
pl_files = glob.glob(PL_DIR + "/*.npz")

model = ExponentialModel() + ConstantModel()
freqs = []
all_res = []
laser_pulses = []
areas = []
for file in pl_files:
    freq_str = os.path.basename(file).split('.')[0]
    freq_str = freq_str[5:]  # remove 'freq_'
    freq_str_decimal = freq_str.replace('_', '.')
    freq = float(freq_str_decimal)
    freqs.append(freq)

    data = np.load(file)
    bins = data['bins'][CUTOFF_IDX:]
    hist = data['hist'][CUTOFF_IDX:]
    laser_pulses.append(data['hist'][0])
    areas.append(np.sum(hist))

    res = model.fit(hist, x=bins,
                    decay=0.01)
    all_res.append(res)

freqs = np.array(freqs)
freq_min = min(freqs)
freqs = freqs - freq_min

# get data from fits
amplitudes = np.fromiter(map(lambda x: x.params['amplitude'].value, all_res), float)
amplitude_err = np.fromiter(map(lambda x: x.params['amplitude'].stderr, all_res), float)
bgs = np.fromiter(map(lambda x: x.params['c'].value, all_res), float)
bg_err = np.fromiter(map(lambda x: x.params['c'].stderr, all_res), float)
tau = np.fromiter(map(lambda x: x.params['decay'].value, all_res), float)
tau_err = np.fromiter(map(lambda x: x.params['decay'].stderr, all_res), float)
area_fit = amplitudes * tau
area_err = area_fit * np.sqrt((amplitude_err / amplitudes) ** 2 + (tau_err / tau) ** 2)

# plotting
fig, axs = plt.subplots(2, 1, figsize=(9, 6),
                        sharex=True)
ax1_r = axs[1].twinx()
# cavity transmission
axs[0].plot(freq_cavity + (FREQ_START - freq_min - AOM_FREQ), transmission,
            color='cornflowerblue')
# PL
axs[1].errorbar(freqs, area_fit, yerr=area_err,
                ls='', marker='o', capsize=3, color='coral')
# T1 lifetime
ax1_r.errorbar(freqs, 1e3*tau, yerr=1e3*tau_err,
               ls='', marker='^', capsize=3, color='mediumpurple')

axs[0].set_title('300 mT PL and Cavity Resonance')
axs[0].set_ylabel('Cavity Reflection (A.U.)')
axs[1].set_ylabel('PL Area (A.U.)', color='coral')
ax1_r.set_ylabel('Fitted PL Lifetime (ms)', color='mediumpurple')
axs[-1].set_xlabel(f'Frequency - {freq_min + AOM_FREQ:.3f} (GHz)')
ax1_r.set_ylim((0, 15))
axs[-1].set_xlim((2, 7))

# plt.tight_layout(rect=(0.07, 0, 1, 1))
plt.tight_layout()
if SAVE_FIG:
    plt.savefig(SAVE_NAME)
else:
    plt.show()
