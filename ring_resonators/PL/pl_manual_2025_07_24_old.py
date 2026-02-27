import glob
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel
import pickle


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Mounted_device_mk_3/4K/2025_07_24/pl")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/mounted_mk_3/10mK_pl/manual_pl_2025_07_24")
CUTOFF_IDX = 2

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')


pl_files = glob.glob('*.npz', root_dir=DATA_DIR)

model = ExponentialModel() + ConstantModel()
for file in pl_files:
    file_base = os.path.splitext(file)[0]
    file_parts = file_base.split('_')
    identifier = file_parts[2]

    data = np.load(os.path.join(DATA_DIR, file))
    bins = data['bins'][CUTOFF_IDX:]
    hist = data['counts'][CUTOFF_IDX:]

    res = model.fit(hist, x=bins,
                    amplitude=10, decay=9, c=10)

    t1 = res.params['decay'].value
    t1_err = res.params['decay'].stderr
    text = rf'$T_1$ = {t1:.3f} $\pm$ {t1_err:.3f} ms'

    plt.plot(bins, hist,
             ls='', marker='o', color='cornflowerblue')
    plt.plot(bins, res.best_fit,
             'k--')
    ax = plt.gca()
    plt.text(0.95, 0.95, text,
             ha='right', va='top',
             transform=ax.transAxes)
    plt.xlabel('Time (ms)')
    plt.ylabel('Counts')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, identifier + '.png'))
    plt.clf()
