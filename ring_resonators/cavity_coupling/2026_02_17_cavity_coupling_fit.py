import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from cav_transmission import R_mod


DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
        '/Mounted_device_mk_5/10mK/2026_02_17/initialization_snspd/snspd_sweep_04.txt')
FREQ_START =194824.456
FREQ_END = 194833.227

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (-4, 4)
color_source = 'cornflowerblue'
color_memory = 'coral'

# fitting guesses
# frequency parameters
w_cav_guess = 5.3
w_ion_guess = 5.4

# cavity parameters
kappa_guess = 0.5
kappa_in_guess = kappa_guess / 2

# coupling parameters
coupling_guess = 0.25  # unit: GHz
inhomog_guess = 0.3  # unit: GHz

# fitting parameters
a_guess = 8000

# read data and fit
df = pd.read_csv(DATA, sep='\t')
transmission = df['counts']
time = df['time(ps)']
freq = np.linspace(0, FREQ_END-FREQ_START, len(time))

# fitting of data
p0 = (w_cav_guess, w_ion_guess, kappa_guess, kappa_in_guess, coupling_guess, inhomog_guess, a_guess)
popt, pcov = curve_fit(R_mod, freq, transmission, p0=p0)
print(popt)

plt.plot(freq, transmission,
         label='Data')
plt.plot(freq, R_mod(freq, *popt),
         ls='--', color='k', label='Fit')
plt.plot(freq, R_mod(freq, popt[0], popt[1], popt[2], popt[3], 0, popt[5], popt[6]),
         ls='--', color='g', label='Fit with No Ions')
# plt.plot(freq, R_mod(freq, *p0),
#          ls='--', color='r', label='Initial Guess')
plt.xlabel(f'Detuning (GHz) from {FREQ_START} GHz')
plt.ylabel('Transmission (A.U.)')
plt.legend()
plt.tight_layout()
plt.show()

# print report
parameters = ['Cavity Center Frequency',
              'Ion Center Frequency',
              'Cavity Linewidth',
              'Cavity Input Coupling',
              'Ensemble Coupling',
              'Inhomogeneous Linewidth',
              'a']
units = ['GHz',
         'GHz',
         'GHz',
         'GHz',
         'GHz',
         'GHz',
         '']

print('Fitted Parameters:')
for value, error, parameter_name, unit in zip(popt, np.diag(pcov), parameters, units):
    print(f'\t{parameter_name}: {value:.3f} +/- {error:.3f} {unit}')

print(f'Fitted Cooperativity: {4*(popt[4]**2)/(popt[2]*popt[5])}')
