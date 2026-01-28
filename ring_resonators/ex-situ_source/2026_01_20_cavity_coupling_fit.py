import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from cav_transmission import R_no_gamma


DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
             '/Unmounted_device_roomtemp/2026_01_20/2CAV.csv')
LASER_OFF_PATH = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                  '/Unmounted_device_roomtemp/2026_01_14/cavity_scan/background/data_000000.csv')
START_FREQ = 194827.613
END_FREQ = 194831.547
REF_FREQ = 194829.540

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
xlim = (-4, 4)
color_source = 'cornflowerblue'
color_memory = 'coral'

# fitting guesses
# frequency parameters
w_cav_guess = 1.7
w_ion_guess = 2.0

# cavity parameters
kappa_guess = 0.4

# coupling parameters
omega_guess = 0.1  # unit: GHz
delta_guess = 0.1  # unit: GHz

# fitting parameters
a_guess = 7
b_guess = 0
phi_guess = 0


# read background data
bg_df = pd.read_csv(LASER_OFF_PATH)
transmission = bg_df['Data Voltage (V)'].astype(float)
bg_avg = np.min(transmission)

df = pd.read_csv(DATA, header=10, skiprows=[11])

ramp = df['CH1'].astype(float).to_numpy()
transmission_memory = df['CH3'].astype(float).to_numpy()

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
ramp = ramp[id_min:id_max]
transmission_memory = transmission_memory[id_min:id_max]
transmission_memory -= bg_avg

# convert time to frequency
freq = np.linspace(0, END_FREQ-START_FREQ, id_max-id_min)  # unit: GHz

# fitting of data
p0 = (w_cav_guess, w_ion_guess, kappa_guess, delta_guess, omega_guess, a_guess, b_guess, phi_guess)
popt, pcov = curve_fit(R_no_gamma, freq, transmission_memory, p0=p0)
print(popt)

r_guess = R_no_gamma(freq, w_cav_guess, w_ion_guess, kappa_guess, delta_guess, omega_guess, a_guess, b_guess, phi_guess)
plt.plot(freq, transmission_memory,
         label='Data')
plt.plot(freq, R_no_gamma(freq, *popt),
         ls='--', color='k', label='Fit')
# plt.plot(freq, r_guess,
#          ls='--', color='r', label='Initial Guess')
plt.xlabel(f'Detuning (GHz) from {START_FREQ} GHz')
plt.ylabel('Transmission (A.U.)')
plt.legend()
plt.tight_layout()
plt.show()

# print report
parameters = ['Cavity Center Frequency',
              'Ion Center Frequency',
              'Cavity Linewidth',
              'Inhomogeneous Linewidth',
              'Ensemble Coupling',
              'a',
              'b',
              'phi']
units = ['GHz',
         'GHz',
         'GHz',
         'GHz',
         'GHz',
         '',
         '',
         '']

print('Fitted Parameters:')
for value, error, parameter_name, unit in zip(popt, np.diag(pcov), parameters, units):
    print(f'\t{parameter_name}: {value:.3f} +/- {error:.3f} {unit}')
