import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel
import pickle


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/PL_scan_2025_03_25")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/10mK_pl/all_fitted_decay/03252025")
CUTOFF_IDX = 5

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')


# make output dir
if not os.path.exists(OUTPUT_DIR):
    print(f'Creating directory {OUTPUT_DIR}')
    os.makedirs(OUTPUT_DIR)

# get data
pl_files = glob.glob(DATA_DIR + "/*181805*.npz")

model = ExponentialModel() + ConstantModel()
freqs = []
all_res = []
laser_pulses = []
areas = []
for file in pl_files:
    data = np.load(file)
    freq = data['freq']
    bins = data['bins'][CUTOFF_IDX:]
    hist = data['hist'][CUTOFF_IDX:]
    laser_pulses.append(data['hist'][0])
    areas.append(np.sum(hist))
    freqs.append(freq)

    res = model.fit(hist, x=bins,
                    decay=0.01, amplitude=10,
                    c=5)
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
    freq_str = f'{freq:.3f}'.replace('.', '_')
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
area_fit = amplitudes * tau
area_err = area_fit * np.sqrt((amplitude_err / amplitudes) ** 2 + (tau_err / tau) ** 2)

# save fit data
save_data = {
    "freqs": freqs,
    "freq_min": freq_min,
    "amplitudes": (amplitudes, amplitude_err),
    "bgs": (bgs, bg_err),
    "tau": (tau, tau_err),
    "area_fit": (area_fit, area_err),
}
save_name_data = f"res_data.bin"
with open(os.path.join(OUTPUT_DIR, save_name_data), "wb") as f:
    pickle.dump(save_data, f)


# plotting of PL
plt.errorbar(freqs, area_fit, area_err,
             ls='', marker='o', capsize=3, color='cornflowerblue')
plt.xlabel(f'Frequency + {freq_min:.3f} (GHz)')
plt.ylabel('Fitted PL Area (A.U.)')
plt.grid(True)

plt.tight_layout()
plt.show()


# plotting of PL
plt.errorbar(freqs, amplitudes, yerr=amplitude_err,
             ls='', marker='o', capsize=3, color='cornflowerblue')
plt.xlabel(f'Frequency + {freq_min:.3f} (GHz)')
plt.ylabel('Fitted PL Amplitude (A.U.)')
plt.grid(True)
plt.ylim((0, 120))

plt.tight_layout()
plt.show()


# plotting of other parameters
fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
axs[0].errorbar(freqs, bgs, yerr=bg_err,
                ls='', marker='o', capsize=3, color='cornflowerblue')
axs[1].plot(freqs, laser_pulses,
            ls='', marker='o', color='coral')
axs[2].errorbar(freqs, 1e3*tau, yerr=1e3*tau_err,
                ls='', marker='o', capsize=3, color='mediumpurple')

axs[0].set_ylabel('Fitted Background (A.U.)')
axs[1].set_ylabel('Laser Counts')
axs[2].set_ylabel('PL Lifetime (ms)')
axs[-1].set_xlabel(f'Frequency + {freq_min:.3f} (GHz)')
# axs[0].set_ylim((0, 5))
axs[2].set_ylim((0, 15))

plt.tight_layout()
plt.show()


# get data on max position
max_idx = np.argmax(amplitudes)
max_freq = freqs[max_idx]
print(f"Frequency (no AOM): {max_freq + freq_min:.3f}")
