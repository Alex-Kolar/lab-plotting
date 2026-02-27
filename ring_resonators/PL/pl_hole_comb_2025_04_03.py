import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/10mK/PL_holeburn_2025_04_03/PL_manual_2025_04_03")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/new_mounted/10mK_pl/winter_2025_cooldown/optical_pumping/afc_04032025")
FILE_NUMBERS = [24, 25, 26, 27]
FILE_FREQS = [-15, -5, 5, 15]  # Detuning in MHz

# data params
integration_time = 300  # unit: s
pl_freq = 20  # unit: Hz
pl_period = (1/pl_freq)*1e3  # unit: ms

# processing params
idx_to_skip = 5
smoothing_fit_cutoff = 10  # unit: s


# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color2 = 'coral'
color3 = 'mediumpurple'
SAVE_FIGS = True  # if true, dump to folder. Else, just plot.


all_res = []
for num in FILE_NUMBERS:
    # get data
    data_file = os.path.join(DATA_DIR, f'holeburn{num}.npy')
    data = np.load(data_file)
    pl_data = np.sum(data, axis=0)
    pl_time = np.linspace(0, pl_period, num=len(pl_data))
    counts_data = np.sum(data, axis=1)
    counts_time = np.linspace(0, integration_time, num=len(counts_data))

    # fit data
    model_pl = ExponentialModel() + ConstantModel()
    res_pl = model_pl.fit(pl_data[idx_to_skip:], x=pl_time[idx_to_skip:],
                          decay=5, amplitude=80)
    all_res.append(res_pl)

    # plot PL
    plt.plot(pl_time[idx_to_skip:], pl_data[idx_to_skip:],
             color=color)
    plt.plot(pl_time[idx_to_skip:], res_pl.best_fit,
             color='k', ls='--')
    # add text for decay
    t1 = res_pl.params['decay'].value
    t1_err = res_pl.params['decay'].stderr
    text = rf'$T_1$ = {t1:.3f} $\pm$ {t1_err:.3f} ms'
    ax = plt.gca()
    plt.text(0.95, 0.95, text,
             ha='right', va='top',
             transform=ax.transAxes)
    plt.title('Photoluminescence Measurement')
    plt.xlabel('Time (ms)')
    plt.ylabel('Counts')
    plt.xlim(0, pl_period)

    plt.tight_layout()
    plt.show()


    # plot hole burning
    plt.plot(counts_time, counts_data,
             color=color2)
    plt.title('Hole Burn Measurement')
    plt.xlabel('Time (s)')
    plt.ylabel('Counts per Excitation Pulse')
    plt.xlim(0, smoothing_fit_cutoff)

    plt.tight_layout()
    plt.show()


# plot fitted params
all_t1 = [res.params['decay'].value for res in all_res]
all_t1_err = [res.params['decay'].stderr for res in all_res]

plt.errorbar(FILE_FREQS, all_t1, yerr=all_t1_err,
             marker='o', ls='', color=color3, capsize=3)
plt.xlabel('Detuning from AFC center (MHz)')
plt.ylabel('PL Lifetime (ms)')

plt.tight_layout()
plt.show()
