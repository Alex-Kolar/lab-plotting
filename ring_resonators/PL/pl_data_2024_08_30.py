import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/pl_08302024")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/10mK_pl/08302024")
CUTOFF_IDX = 5

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')


pl_files = glob.glob(DATA_DIR + "/*.npz")

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
    plt.ylim((0, 60))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR + '/' + freq_str + '.png')
    plt.clf()

freqs = np.array(freqs)
freq_min = min(freqs)
freqs = freqs - freq_min

# get data from fits
amplitudes = list(map(lambda x: x.params['amplitude'].value, all_res))
amplitude_err = list(map(lambda x: x.params['amplitude'].stderr, all_res))
bgs = list(map(lambda x: x.params['c'].value, all_res))
bg_err = list(map(lambda x: x.params['c'].stderr, all_res))


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
