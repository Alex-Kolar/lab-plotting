import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel
import pickle


DATA_NO_PUMP = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
                "/New_mounted_device/10mK/PL_scan_2024_10_28/freq_194821_386474_pump_time_0.0.npz")
DATA_PUMP = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
             "/New_mounted_device/10mK/PL_scan_2024_10_28/freq_194821_38640900006_pump_time_2.0.npz")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/10mK_pl/all_fitted_decay/10282024_test")
CUTOFF_IDX = 5

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')


model = ExponentialModel() + ConstantModel()
all_res = []
laser_pulses = []
areas = []
for file in (DATA_NO_PUMP, DATA_PUMP):
    file_parts = file.split('_')
    pump_str = file_parts[-1]
    pump_str = pump_str[:-4]  # remove '.npz'
    pump_str = pump_str.replace('.', '_')

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
    plt.ylim((0, 40))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + '/' + pump_str + '.png')
    plt.clf()

# get data from fits
amplitudes = np.fromiter(map(lambda x: x.params['amplitude'].value, all_res), float)
amplitude_err = np.fromiter(map(lambda x: x.params['amplitude'].stderr, all_res), float)
bgs = np.fromiter(map(lambda x: x.params['c'].value, all_res), float)
bg_err = np.fromiter(map(lambda x: x.params['c'].stderr, all_res), float)
tau = np.fromiter(map(lambda x: x.params['decay'].value, all_res), float)
tau_err = np.fromiter(map(lambda x: x.params['decay'].stderr, all_res), float)
area_fit = amplitudes * tau
area_err = area_fit * np.sqrt((amplitude_err / amplitudes) ** 2 + (tau_err / tau) ** 2)

# save fit data
save_data = {
    "amplitudes": (amplitudes, amplitude_err),
    "bgs": (bgs, bg_err),
    "tau": (tau, tau_err),
    "area_fit": (area_fit, area_err),
}
save_name_data = f"res_data.bin"
with open(os.path.join(OUTPUT_DIR, save_name_data), "wb") as f:
    pickle.dump(save_data, f)


# plotting of PL
x_points = np.arange(len(all_res))
plt.errorbar(x_points, area_fit, area_err,
             ls='', marker='', capsize=3, color='k', zorder=4)
plt.bar(x_points, area_fit,
        color='cornflowerblue', edgecolor='k', zorder=3)
plt.ylabel('Fitted PL Area (A.U.)')
plt.grid(True, axis='y')
plt.xticks(x_points, ["No Pump", "2 s Pump"])

plt.tight_layout()
plt.show()


# plotting of other parameters
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
axs[0].errorbar(x_points, bgs, yerr=bg_err,
                ls='', marker='', capsize=3, color='k')
axs[0].bar(x_points, bgs,
           color='cornflowerblue', edgecolor='k')
axs[1].bar(x_points, laser_pulses,
           color='coral', edgecolor='k')
axs[2].errorbar(x_points, 1e3*tau, yerr=1e3*tau_err,
                ls='', marker='', capsize=3, color='k')
axs[2].bar(x_points, 1e3*tau, yerr=1e3*tau_err,
           color='mediumpurple', edgecolor='k')

axs[0].set_ylabel('Fitted Background (A.U.)')
axs[1].set_ylabel('Laser Counts')
axs[2].set_ylabel('PL Lifetime (ms)')
axs[-1].set_xticks(x_points, ["No Pump", "2 s Pump"])
# axs[0].set_ylim((0, 5))
# axs[2].set_ylim((0, 15))

plt.tight_layout()
plt.show()


print("Ratio of Areas to Laser:",
      area_fit / laser_pulses)
print("Ratio of Backgrounds to Laser:",
      bgs / laser_pulses)
