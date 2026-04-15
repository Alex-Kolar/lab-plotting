import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model, Parameters
from lmfit.models import ConstantModel, BreitWignerModel

from cav_transmission import R_mod_lmfit


DATA = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
        '/Mounted_device_mk_5/10mK/2026_04_14/cavity_field_scan/cavity_scan_1238mT.txt')
DATA_NO_IONS = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                '/Mounted_device_mk_5/10mK/2026_04_14/cavity_field_scan/cavity_scan_0598mT.txt')
FREQ_START = 194824.878
FREQ_END = 194833.854

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
# XLIM = (2, 7)  # unit: GHz
XLIM_range = 6  # unit: GHz
color_ion = 'coral'
color_no_ion = 'cornflowerblue'

# fitting guesses
# frequency parameters
w_cav_guess = 4.2
w_ion_guess = 4.3

# cavity parameters
kappa_guess = 0.433

# coupling parameters
coupling_guess = 0.25  # unit: GHz
inhomog_guess = 0.3  # unit: GHz


# read data and fit
df = pd.read_csv(DATA, sep='\t')
transmission = df['counts']
time = df['time(ps)']
freq = np.linspace(0, FREQ_END-FREQ_START, len(time))

df_no_ions = pd.read_csv(DATA_NO_IONS, sep='\t')
transmission_no_ions = df_no_ions['counts']

# fitting of data

# fit no ion cavity to extract cavity data
model_cavity = BreitWignerModel() + ConstantModel()
res_cavity = model_cavity.fit(transmission_no_ions, x=freq,
                              center=w_cav_guess,
                              sigma=kappa_guess,
                              q=0,
                              amplitude=max(transmission_no_ions))
fitted_kappa = res_cavity.params['sigma'].value
fitted_w_cav = res_cavity.params['center'].value
kappa_in_guess = fitted_kappa / 2
reflection = res_cavity.params['c'].value / (res_cavity.params['c'].value + res_cavity.params['amplitude'].value)
fitted_kappa_in = (fitted_kappa / 2) * (1 + np.sqrt(reflection))
print(f"Fitted kappa: {fitted_kappa:.3f} GHz")
print(f"Fitted w_cav: {fitted_w_cav:.3f} GHz")
print(f"Reflection: {reflection:.3f}")
print(f"Fitted kappa_in: {fitted_kappa_in:.3f} GHz")

model = Model(R_mod_lmfit)
params = Parameters()
params.add('w_cav', value=fitted_w_cav)
params.add('w_ions', value=w_ion_guess)
params.add('kappa', value=fitted_kappa, vary=False)
params.add('kappa_in', value=fitted_kappa_in, vary=False)
params.add('coupling', value=coupling_guess)
params.add('inhomog', value=inhomog_guess)
params.add('a', value=max(transmission))
res = model.fit(transmission, params, x=freq)

# plotting
norm_factor_no_ions = res_cavity.params['amplitude'].value + res_cavity.params['c'].value
norm_factor = res.params['a'].value
freq_centered_no_ions = freq - res_cavity.params['center'].value
freq_centered = freq - res.params['w_cav'].value

plt.plot(freq_centered_no_ions, transmission_no_ions/norm_factor_no_ions,
         color=color_no_ion, label='Cavity Only')
plt.plot(freq_centered_no_ions, res_cavity.best_fit/norm_factor_no_ions,
         ls='--', color='k')
plt.plot(freq_centered, transmission/norm_factor,
         color=color_ion, label='Cavity and Ions')
plt.plot(freq_centered, res.best_fit/norm_factor,
         ls=':', color='k')
plt.xlabel(f'Cavity Detuning (GHZ)')
plt.ylabel('Cavity Reflection (Normalized)')
plt.title('Cavity Coupling Fit')
plt.xlim(-XLIM_range/2, XLIM_range/2)
# plt.ylim(0, 30000)
plt.legend()
plt.tight_layout()
plt.show()

# print reports
print(res_cavity.fit_report())
print(res.fit_report())

coupling = res.params['coupling'].value
coupling_err = res.params['coupling'].stderr
inhomog = res.params['inhomog'].value
inhomog_err = res.params['inhomog'].stderr
kappa = res_cavity.params['sigma'].value
kappa_err = res_cavity.params['sigma'].stderr

coop = (4 * (coupling ** 2)) / (kappa * inhomog)
coop_err = coop * np.sqrt((2*coupling_err/coupling)**2 + (inhomog_err/inhomog)**2 + (kappa_err/kappa)**2)
print(f'Fitted cooperativity: {coop:.3f} +/- {coop_err:.3f}')
