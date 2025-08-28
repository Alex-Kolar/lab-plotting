import os
import pandas as pd
import numpy as np
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel


DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_4/10mK/2025_08_22')
OUTPUT_DIR = ('/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators'
              '/mounted_mk_4/10mK_cavity/10mK_snspd_08222025')
FREQ_START = 194810.775  # unit: GHz
FREQ_END = 194819.611

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})


# load files
files = glob.glob('*.txt', root_dir=DATA_DIR)
files = sorted(files)

# process files
model = BreitWignerModel() + LinearModel()
all_fields = []
all_res = []
for filename in files:
    B_field = int(filename[-9:-6])
    all_fields.append(B_field)

    df = pd.read_csv(os.path.join(DATA_DIR, filename), sep='\t')

    time = df['time(ps)']
    counts = df['counts']
    freq = np.linspace(0, FREQ_END - FREQ_START, len(time))
    res = model.fit(counts, x=freq,
                    sigma=0.3,
                    center=5,
                    amplitude=1200,
                    q=0)
    all_res.append(res)

    plt.plot(freq, counts,
             label='Data')
    # plt.plot(freq, res.init_fit,
    #          ls='--', color='r', label='Initial Fit')
    plt.plot(freq, res.best_fit,
             ls='--', color='k', label='Best Fit')
    plt.title(f'Cavity Scan {B_field} mT')
    plt.xlabel(f'Detuning (GHz) from {FREQ_START:.3f} GHz')
    plt.ylabel('Counts')
    plt.legend()
    plt.tight_layout()

    output_name = f'{B_field}_mT_scan.png'
    plt.savefig(os.path.join(OUTPUT_DIR, output_name))
    plt.clf()

all_lw = np.array([res.params['sigma'].value for res in all_res])
all_lw_err = np.array([res.params['sigma'].stderr for res in all_res])
all_lw *= 1e3  # convert to MHz
all_lw_err *= 1e3

all_contrasts = []
all_contrasts_err = []
for res in all_res:
    slope = res.params['slope'].value
    intercept = res.params['intercept'].value
    center = res.params['center'].value
    amplitude = res.params['amplitude'].value
    constant = slope * center + intercept
    contrast = amplitude / (amplitude + constant)
    all_contrasts.append(contrast)

fig, axs = plt.subplots(2, 1, sharex=True)
axs[0].errorbar(all_fields, all_lw, yerr=all_lw_err,
                color='cornflowerblue', marker='o', ls='', capsize=3)
axs[1].errorbar(all_fields, all_contrasts,
                color='coral', marker='o', ls='', capsize=3)
axs[0].set_title('Coupling Scan')
axs[0].set_ylabel('Cavity Linewidth (MHz)')
axs[1].set_ylabel('Cavity Contrast')
axs[-1].set_xlabel('Magnetic Field (mT)')

fig.tight_layout()
fig.show()
