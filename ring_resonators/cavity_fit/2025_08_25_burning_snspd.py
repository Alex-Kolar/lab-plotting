import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel
from scipy.signal import find_peaks


DATA_PRE = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_4/10mK/2025_08_25/cavity_scan_5min_preburn.txt')
DATA_POST = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
             '/Mounted_device_mk_4/10mK/2025_08_25/cavity_scan_5min_postburn.txt')
FREQ_START = 194810.304  # unit: GHz
FREQ_END = 194818.848


# load files
df_pre = pd.read_csv(DATA_PRE, sep='\t')
df_post = pd.read_csv(DATA_POST, sep='\t')

time = df_pre['time(ps)']
freq = np.linspace(0, FREQ_END-FREQ_START, len(time))
counts_pre = df_pre['counts']
counts_post = df_post['counts']

model = BreitWignerModel() + LinearModel()
param_guesses = {'sigma': 0.3, 'center': 4, 'amplitude': 1200, 'q': 0}
res_pre = model.fit(counts_pre, x=freq, **param_guesses)
res_post = model.fit(counts_post, x=freq, **param_guesses)

print(f'kappa before burning: {res_pre.params['sigma']*1e3:.3f} +/- {res_pre.params['sigma'].stderr*1e3} MHz')
print(f'kappa after burning: {res_post.params['sigma']*1e3:.3f} +/- {res_post.params['sigma'].stderr*1e3} MHz')


# plt.plot(freq, df_pre['counts'])
# plt.plot(freq, df_post['counts'])
# plt.xlim((4, 6))
# plt.xlabel(f'Detuning (GHz) from {FREQ_START:.3f} GHz')
# plt.ylabel('Counts')
#
# plt.show()
