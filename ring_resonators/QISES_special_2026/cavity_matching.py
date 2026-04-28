import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import c
from lmfit.models import BreitWignerModel, ConstantModel
from lmfit import Model, Parameters


# data params
DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Unmounted_device_mk_3/2026_04_15/cavity_scan/device_36')
CSV_PATH = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Unmounted_device_mk_3/2026_04_15/cavity_scan/device_36/resonance_freq_data.csv')
LASER_OFF_PATH = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                  '/Unmounted_device_mk_3/2026_04_15/cavity_scan/device_36/LASEROFF.csv')
signal_file = 29

DATA_COUPLING = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                 '/Mounted_device_mk_5/10mK/2026_04_14/cavity_field_scan/cavity_scan_1238mT.txt')
DATA_NO_IONS = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                '/Mounted_device_mk_5/10mK/2026_04_14/cavity_field_scan/cavity_scan_0598mT.txt')
FREQ_START = 194824.878  # for coupling
FREQ_END = 194833.854

center_guess = 15
amp_guess = 2
c_guess = 0.5
q_guess = 0

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
xlim_range = (-2, 2)


def resonance_helper(freq, trans, center=center_guess, amplitude=amp_guess, c=c_guess, q=q_guess, print_report=False):
    ref_freq = min(freq)
    freq -= ref_freq  # set to detuning in THz
    freq *= 1e3  # convert to GHz
    ref_freq *= 1e3
    model = BreitWignerModel() + ConstantModel()
    res = model.fit(trans, x=freq,
                    center=center, amplitude=amplitude, c=c, q=q)
    # set freq to be detuning from cavity center
    freq -= res.params['center'].value

    if print_report:
        print(res.fit_report())
        cavity_kappa = res.params['sigma'].value
        cavity_freq = ref_freq + res.params['center'].value
        cavity_q = cavity_freq / cavity_kappa
        print(f'Cavity kappa: {cavity_kappa:.3f} GHz')
        print(f'Cavity freq: {cavity_freq:.3f} GHz')
        print(f'Cavity q: {cavity_q:.3f}')

    return res, freq


# read laser offres data
laser_off_df = pd.read_csv(LASER_OFF_PATH, header=10, skiprows=[11])
zero_level = np.mean(laser_off_df['CH1'].astype(float))

# read main csv
main_df = pd.read_csv(CSV_PATH)
idx = np.where(main_df['File'] == signal_file)[0][0]
min_freq = main_df['Minimum (GHz)'][idx]
max_freq = main_df['Maximum (GHz)'][idx]
data_path = os.path.join(DATA_DIR, f'data_{signal_file:06}.csv')
data_df = pd.read_csv(data_path)

# plot signal cavity
ramp = data_df['Ramp Voltage (V)'].astype(float)
transmission = data_df['Data Voltage (V)'].astype(float)
transmission -= zero_level

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
transmission = transmission[id_min:id_max]
transmission.reset_index(drop=True, inplace=True)
freq = np.linspace(min_freq/1e3, max_freq/1e3,
                   num=(id_max - id_min))  # unit: THz
wl = c / freq
wl /= 1e3  # convert to nm

res, freq_detuning = resonance_helper(freq, transmission, print_report=True)
norm_factor = res.params['amplitude'].value + res.params['c'].value

fig, ax = plt.subplots(figsize=(5, 2), dpi=300)
ax.plot(freq_detuning, transmission/norm_factor, color=color)
# ax.plot(freq_detuning, res.best_fit, '--k')
ax.set_xlim(xlim_range)
ax.set_title(f'Pair Source Signal Resonance')
ax.set_xlabel('Cavity Detuning (GHz)')
ax.set_ylabel('Transmission (Norm.)')

fig.tight_layout()
fig.show()

transmission_signal = transmission/norm_factor


# plot bonded cavity
# fitting guesses
# frequency parameters
w_cav_guess = 4.2
w_ion_guess = 4.3

# cavity parameters
kappa_guess = 0.433

# coupling parameters
coupling_guess = 0.25  # unit: GHz
inhomog_guess = 0.3  # unit: GHz


# fitting function
def R_mod_lmfit(x, w_cav, w_ions, kappa, kappa_in, coupling, inhomog, a):
    ion_term = (coupling ** 2) / (x - w_ions + 1j * inhomog / 2)
    t = 1 - (1j*kappa_in)/(x - w_cav + 1j * kappa / 2 - ion_term)
    return a * (np.abs(t) ** 2)


# read data and fit
df = pd.read_csv(DATA_COUPLING, sep='\t')
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

fig, ax = plt.subplots(figsize=(5, 2), dpi=400)
plt.plot(freq_centered, transmission/norm_factor,
         color='coral', label='Coupled Cavity')
plt.xlabel(f'Cavity Detuning (GHZ)')
plt.ylabel('Transmission (Norm.)')
plt.title('Coupled Memory Cavity')
plt.xlim(xlim_range)
plt.tight_layout()
plt.show()


fig, axs = plt.subplots(2, 1, figsize=(5, 4), dpi=400, sharex=True)
axs[0].plot(freq_detuning, transmission_signal,
            color='cornflowerblue')
axs[1].plot(freq_centered, transmission/norm_factor,
            color='coral')
axs[0].set_ylabel('Transmission (Norm.)')
axs[1].set_ylabel('Transmission (Norm.)')
axs[0].set_title('Signal Cavity')
axs[1].set_title('Coupled Memory Cavity')
axs[1].set_xlabel('Cavity Detuning (GHz)')
axs[1].set_xlim(xlim_range)

fig.tight_layout()
fig.show()
