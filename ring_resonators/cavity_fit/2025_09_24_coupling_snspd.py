import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel
from scipy.signal import find_peaks


DATA_OFF = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Mounted_device_mk_4/10mK/2025_09_24/cavity_scan_offres_10min2.txt')
DATA_ON = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
           '/Mounted_device_mk_4/10mK/2025_09_24/cavity_scan_onres_10min2.txt')
FREQ_START = 194821.3404  # unit: GHz
FREQ_END = 194829.7824


# load files
df_on = pd.read_csv(DATA_ON, sep='\t')
df_off = pd.read_csv(DATA_OFF, sep='\t')

time = df_off['time(ps)']
freq = np.linspace(0, FREQ_END-FREQ_START, len(time))

# fitting
model = BreitWignerModel() + LinearModel()
counts_off = df_off['counts']
counts_on = df_on['counts']

res_off = model.fit(counts_off, x=freq,
                    center=4.5, amplitude=1000)
res_on = model.fit(counts_on, x=freq,
                   center=4.5, amplitude=600)
print(f'Linewidth off resonant: {res_off.params['sigma'].value*1e3} MHz')
print(f'Linewidth on resonant: {res_on.params['sigma'].value*1e3} MHz')

center = res_off.params['center'].value
amplitude = res_off.params['amplitude'].value
slope = res_off.params['slope'].value
intercept = res_off.params['intercept'].value
constant = slope * center + intercept
contrast = amplitude / (constant + amplitude)
print(f'Contrast off resonant: {contrast}')

center = res_on.params['center'].value
amplitude = res_on.params['amplitude'].value
slope = res_on.params['slope'].value
intercept = res_on.params['intercept'].value
constant = slope * center + intercept
contrast = amplitude / (constant + amplitude)
print(f'Contrast on resonant: {contrast}')


# plot fit for off resonant
plt.plot(freq, counts_off,
         label='Data')
plt.plot(freq, res_off.best_fit,
         ls='--', color='k', label='Fit')
plt.title('Off-Resonant Fit')
plt.tight_layout()
plt.show()

# plot fit for on resonant
plt.plot(freq, counts_on,
         label='Data')
plt.plot(freq, res_on.best_fit,
         ls='--', color='k', label='Fit')
plt.title('On-Resonant Fit')
plt.tight_layout()
plt.show()

# plot both
plt.plot(freq, counts_off,
         label='Off-resonant')
plt.plot(freq, counts_on,
         label='On-resonant')
# plt.xlim((4, 6))
plt.xlabel(f'Detuning (GHz) from {FREQ_START:.3f} GHz')
plt.ylabel('Counts')
plt.legend()
plt.tight_layout()
plt.show()
