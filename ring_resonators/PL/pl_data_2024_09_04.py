import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/pl_09042024")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/10mK_pl/09042024")
OUTPUT_DIR_FINAL = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs"
                    "/ring_resonators/new_mounted/10mK_pl/600mT_area_AOM_orange.svg")
AOM_FREQ = 0.6  # unit: GHz

CUTOFF_IDX = 5

START_SWEEP = 0.972  # unit: GHz
END_SWEEP = 1.226  # unit: GHz

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')
SAVE_FITS = False
SAVE_FIG = True


pl_files = glob.glob(DATA_DIR + "/*.npz")

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

    if SAVE_FITS:
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
        # plt.ylim((0, 100))
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


# plotting of PL
plt.errorbar(freqs, areas,
             ls='', marker='o', capsize=3, color='coral')
plt.title('600 mT PL')
plt.xlabel(f'Frequency + {freq_min + AOM_FREQ:.3f} (GHz)')
plt.ylabel('PL Area (A.U.)')
plt.grid(True)

plt.tight_layout()
if SAVE_FIG:
    plt.savefig(OUTPUT_DIR_FINAL)
else:
    plt.show()
plt.clf()


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
axs[-1].set_xlabel(f'Frequency + {freq_min + AOM_FREQ:.3f} (GHz)')
axs[0].set_ylim((0, 20))
axs[2].set_ylim((0, 15))

plt.tight_layout()
plt.show()

# plotting of pl with sweep area
max_idx: int = np.argmax(areas)
max_freq = freqs[max_idx]
print(f"Start laser frequency: {freq_min + AOM_FREQ + max_freq + START_SWEEP} GHz")
print(f"Stop laser frequency: {freq_min + AOM_FREQ + max_freq + END_SWEEP} GHz")
print(f"Center laser frequency: {freq_min + AOM_FREQ + max_freq + (START_SWEEP + END_SWEEP)/2} GHz")

plt.errorbar(freqs, areas,
             ls='', marker='o', capsize=3, color='cornflowerblue')
plt.axvline(max_freq, color='k', ls='--')
plt.axvline(max_freq + START_SWEEP, color='r', ls='--')
plt.axvline(max_freq + END_SWEEP, color='r', ls='--')
plt.text(x=max_freq, y=min(areas),
         s=f'{freq_min + AOM_FREQ + max_freq:.3f} GHz')
plt.xlabel(f'Frequency + {freq_min + AOM_FREQ:.3f} (GHz)')
plt.ylabel('PL Area (A.U.)')
plt.grid(True)

plt.tight_layout()
plt.show()
