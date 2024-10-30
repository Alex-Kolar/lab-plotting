import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/PL_scan_2024_10_29")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/10mK_pl/all_fitted_decay/10292024/cutoff_20")
CUTOFF_IDX = 20

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')


pl_files = glob.glob(DATA_DIR + "/*.npz")

model = ExponentialModel() + ConstantModel()
pump_times = []
all_res = []
laser_pulses = []
areas = []
for file in pl_files:
    pump_str = file.split('_')[-1]
    pump_str = pump_str[:-4]  # remove '.npz'
    pump_str_save = pump_str.replace('.', '_')
    pump_time = float(pump_str)
    pump_times.append(pump_time)

    data = np.load(file)
    bins = data['bins'][CUTOFF_IDX:]
    hist = data['hist'][CUTOFF_IDX:]
    laser_pulses.append(data['hist'][0])
    areas.append(np.sum(hist))

    res = model.fit(hist, x=bins,
                    decay=0.006, c=10)
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
    # plt.ylim((0, 40))
    # plt.yscale('log')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + '/' + pump_str_save + '.png')
    plt.clf()

pump_times = np.array(pump_times)

# get data from fits
amplitudes = np.fromiter(map(lambda x: x.params['amplitude'].value, all_res), float)
amplitude_err = np.fromiter(map(lambda x: x.params['amplitude'].stderr, all_res), float)
bgs = np.fromiter(map(lambda x: x.params['c'].value, all_res), float)
bg_err = np.fromiter(map(lambda x: x.params['c'].stderr, all_res), float)
tau = np.fromiter(map(lambda x: x.params['decay'].value, all_res), float)
tau_err = np.fromiter(map(lambda x: x.params['decay'].stderr, all_res), float)
area_fit = amplitudes * tau
area_err = area_fit * np.sqrt((amplitude_err / amplitudes) ** 2 + (tau_err / tau) ** 2)


# plotting of PL
plt.errorbar(pump_times, area_fit, yerr=area_err,
             ls='', marker='o', capsize=3, color='cornflowerblue')
plt.xlabel(f'Pump Time (s)')
plt.ylabel('Fitted PL Area (A.U.)')
# plt.grid(True)

plt.tight_layout()
plt.show()


# plotting of PL
plt.errorbar(pump_times, amplitudes, yerr=amplitude_err,
             ls='', marker='o', capsize=3, color='cornflowerblue')
plt.xlabel(f'Pump Time (s)')
plt.ylabel('Fitted PL Amplitude (A.U.)')
# plt.grid(True)

plt.tight_layout()
plt.show()


# plotting of other parameters
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
axs[0].errorbar(pump_times, bgs, yerr=bg_err,
                ls='', marker='o', capsize=3, color='cornflowerblue')
axs[1].plot(pump_times, laser_pulses,
            ls='', marker='o', color='coral')
axs[2].errorbar(pump_times, 1e3*tau, yerr=1e3*tau_err,
                ls='', marker='o', capsize=3, color='mediumpurple')

axs[0].set_ylabel('Fitted Background (A.U.)')
axs[1].set_ylabel('Laser Counts')
axs[2].set_ylabel('PL Lifetime (ms)')
axs[-1].set_xlabel(f'Pump Time (s)')
# axs[2].set_ylim((0, 15))

plt.tight_layout()
plt.show()
