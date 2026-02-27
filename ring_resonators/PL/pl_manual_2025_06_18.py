import glob
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel


DATA_ON = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/New_mounted_device/10mK/PL_manual_2025_06_18/Histogram_PL_194833_546.txt")
DATA_OFF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/New_mounted_device/10mK/PL_manual_2025_06_18/Histogram_PL_194834_531.txt")
CUTOFF_IDX = 5

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')


data_on = pd.read_csv(DATA_ON, sep='\t')
data_off = pd.read_csv(DATA_OFF, sep='\t')
time = data_on['time(ps)'][CUTOFF_IDX:] / 1e9  # convert to ms

# fitting
model = ExponentialModel() + ConstantModel()
res_on = model.fit(data_on['counts'][CUTOFF_IDX:], x=time)
res_off = model.fit(data_off['counts'][CUTOFF_IDX:], x=time)
print(res_on.fit_report())
print(res_off.fit_report())


plt.plot(time, data_on['counts'][CUTOFF_IDX:])
plt.plot(time, res_on.best_fit,
         color='k', ls='--')
ax = plt.gca()
text = rf'$T_1$ = {res_on.params['decay'].value:.3f} $\pm$ {res_on.params['decay'].stderr:.3f} ms'
plt.text(0.95, 0.95, text,
         ha='right', va='top',
         transform=ax.transAxes)
plt.title('On-Resonant PL')
plt.xlabel('Time (ms)')
plt.ylabel('Counts')
plt.tight_layout()
plt.show()

plt.plot(time, data_off['counts'][CUTOFF_IDX:])
plt.plot(time, res_off.best_fit,
         color='k', ls='--')
ax = plt.gca()
text = rf'$T_1$ = {res_off.params['decay'].value:.3f} $\pm$ {res_off.params['decay'].stderr:.3f} ms'
plt.text(0.95, 0.95, text,
         ha='right', va='top',
         transform=ax.transAxes)
plt.title('Off-Resonant PL')
plt.xlabel('Time (ms)')
plt.ylabel('Counts')
plt.tight_layout()
plt.show()
