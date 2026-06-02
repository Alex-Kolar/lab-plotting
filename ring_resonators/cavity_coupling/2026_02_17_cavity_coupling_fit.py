import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model, Parameters

from cav_transmission import R_mod_lmfit


DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
        '/Mounted_device_mk_5/10mK/2026_02_17/initialization_snspd/snspd_sweep_04.txt')
FREQ_START = 194824.456
FREQ_END = 194833.227

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
XLIM = (2, 8)  # unit: GHz
color = 'cornflowerblue'

# fitting guesses
# frequency parameters
w_cav_guess = 5.3
w_ion_guess = 5.4

# cavity parameters
kappa_guess = 0.433  # constrained
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
print(len(time))
print(time[1] - time[0])
freq = np.linspace(0, FREQ_END-FREQ_START, len(time))

# fitting of data
model = Model(R_mod_lmfit)
params = Parameters()
params.add('w_cav', value=w_cav_guess)
params.add('w_ions', value=w_ion_guess)
params.add('kappa', value=kappa_guess, vary=False)
params.add('kappa_in', value=kappa_in_guess)
params.add('coupling', value=coupling_guess)
params.add('inhomog', value=inhomog_guess)
params.add('a', value=a_guess)
res = model.fit(transmission, params, x=freq)


plt.plot(freq, transmission,
         color=color, label='Data')
plt.plot(freq, res.best_fit,
         ls='--', color='k', label='Fit')
plt.xlabel(f'Frequency - {FREQ_START} GHz')
plt.ylabel('Transmission (A.U.)')
plt.title('Cavity Coupling Fit')
plt.xlim(XLIM)
plt.ylim(0, 9000)
plt.legend()
plt.tight_layout()
plt.show()

# print report
print(res.fit_report())

coupling = res.params['coupling'].value
inhomog = res.params['inhomog'].value
kappa = res.params['kappa'].value

coop = (4 * (coupling ** 2)) / (kappa * inhomog)
print(f'Fitted cooperativity: {coop:.3f}')
