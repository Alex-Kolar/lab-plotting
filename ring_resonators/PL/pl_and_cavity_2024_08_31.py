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
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/10mK_pl/08312024_combined")
FREQ_START = 194811.486  # unit: GHz
FREQ_END = 194819.973  # unit: GHz
AOM_FREQ = 0.6  # unit: GHz

CUTOFF_IDX = 5

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')


# get cavity data
df = pd.read_csv(CAVITY_DIR, header=10, skiprows=[11])
ramp = df['CH1'].astype(float).to_numpy()
transmission = df['CH2'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission = transmission[id_min:id_max]

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

    t1 = res.params['decay'].value
    t1_err = res.params['decay'].stderr
    text = rf'$T_1$ = {t1*1e3:.3f} $\pm$ {t1_err*1e3:.3f} ms'

    plt.plot(bins, hist,
             ls='', marker='o', color='cornflowerblue')
    plt.plot(bins, res.best_fit,
             'k--')
    ax = plt.gca()
    plt.text(0.95, 0.95, text,
             ha='right', va='top',
             transform=ax.transAxes)
    plt.xlabel('Time (s)')
    plt.ylabel('Counts')
    plt.ylim((0, 100))
    # plt.yscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + '/' + freq_str + '.png')
    plt.clf()

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


# plotting
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
# cavity transmission
axs[0].plot(freq_cavity + (FREQ_START - freq_min - AOM_FREQ), transmission,
            color='cornflowerblue')
# PL
axs[1].plot(freqs, areas,
            ls='', marker='o', color='coral')
# T1 lifetime
axs[2].errorbar(freqs, 1e3*tau, yerr=1e3*tau_err,
                ls='', marker='o', capsize=3, color='mediumpurple')

axs[0].set_ylabel('Cavity Transmission (A.U.)')
axs[1].set_ylabel('PL Area (A.U.)')
axs[2].set_ylabel('PL Lifetime (ms)')
axs[-1].set_xlabel(f'Frequency + {freq_min + AOM_FREQ:.3f} (GHz)')
axs[2].set_ylim((0, 15))
axs[0].grid(True)
axs[1].grid(True)
axs[2].grid(True)

plt.tight_layout()
plt.show()
