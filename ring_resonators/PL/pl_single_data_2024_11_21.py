import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import ExponentialModel, ConstantModel
import pickle


DATA_ON = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
           "/Bulk_crystal/PL_2024_11_21/freq_196041_97395400002_low_atten.npz")
DATA_OFF = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Bulk_crystal/PL_2024_11_21/freq_196044_47215100002_high_atten_reflect.npz")
CUTOFF_IDX = 5

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')


# fit bulk crystal data
data = np.load(DATA_ON)
bins = data['bins'][CUTOFF_IDX:]
hist = data['hist'][CUTOFF_IDX:]

model = ExponentialModel() + ConstantModel()
res = model.fit(hist, x=bins,
                decay=0.01, c=30)

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
plt.title('Erbium PL Measurement')
plt.xlabel('Time (s)')
plt.ylabel('Counts')
plt.ylim((0, 100))

plt.tight_layout()
plt.show()


# fit retroreflector data
data = np.load(DATA_OFF)
bins = data['bins'][CUTOFF_IDX:]
hist = data['hist'][CUTOFF_IDX:]

model = ConstantModel()
res = model.fit(hist, x=bins,
                c=30)

plt.plot(bins, hist,
         ls='', marker='o', color='coral')
plt.plot(bins, res.best_fit,
         'k--')
plt.title('Retroreflector PL Measurement')
plt.xlabel('Time (s)')
plt.ylabel('Counts')
plt.ylim((0, 100))

plt.tight_layout()
plt.show()
