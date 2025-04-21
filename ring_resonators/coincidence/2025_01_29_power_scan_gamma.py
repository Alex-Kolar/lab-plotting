import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit import Model
from lmfit.models import ConstantModel, BreitWignerModel

from ring_resonators.cavity_fit.cavity_metrics import *


# coincidence data
DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/New_mounted_device/300K_no_erbium/coincidence/01292025_pair_gen_power_scan")
integration_time = 5 * 60  # units: s

# resonance data for extracting cavity parameters
CAVITY_DATA = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
               "/New_mounted_device/300K_no_erbium/01162025/SDS00013.csv")
SCAN_RANGE = (195010.481, 195014.446)

# parameters for the device (to use for gamma calculation)
n_eff = 2.18
L = np.pi * 440e-6  # unit: m
c = 3e8  # unit: m/s

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color_coincidence = 'coral'
PLOT_ALL_COINCIDENCE = False  # plot all fitted coincidence curves


# efficiencies
# input laser light
efficiency_laser_coupling = 0.5
efficiency_spectrometer = 0.5
efficiency_chip_coupling = 0.05413
total_efficiency = efficiency_laser_coupling * efficiency_spectrometer * efficiency_chip_coupling
# efficiencies for output light
efficiency_52 = efficiency_chip_coupling * 0.7274 * 0.4532  # path taken by photons at 195.2 THz
efficiency_48 = efficiency_chip_coupling * 0.4790 * 0.9260  # path taken by photons at 194.8 THz
efficiency_pairs = efficiency_48 * efficiency_52


# fitting function versus power
def quadratic(x, a):
    return a * (x ** 2)


# gather data for coincidences
coincidence_files = sorted(glob.glob('pairs_*', root_dir=DATA_DIR))
dfs_coinc = []  # store all dataframes
powers = []  # store all pump powers
int_times = []  # store all integration times
all_res = []  # store all fit result objects

# set up model for fitting
model = Model(g_2_no_delta)

for filename in coincidence_files:
    file_str = os.path.splitext(filename)[0]
    file_path = os.path.join(DATA_DIR, filename)

    # get metadata from file
    file_info = file_str.split('_')  # 'pairs', XXmW, Xmin
    power = float(file_info[1][:2])
    int_time = float(file_info[2][:1])
    power *= total_efficiency
    powers.append(power)
    int_times.append(int_time)

    # read coincidence data
    df = pd.read_csv(file_path, sep='\t')
    dfs_coinc.append(df)
    time = df['Time(ps)'] * 1e-3  # convert to ns
    coincidences = df['Counts']

    # fitting
    x0_guess = time[np.argmax(coincidences)]
    amplitude_guess = np.max(coincidences)
    kappa_guess = 1
    g_guess = 0.125
    res = model.fit(coincidences, x=time,
                    x0=x0_guess,
                    amplitude=amplitude_guess,
                    kappa=kappa_guess,
                    g=g_guess)
    all_res.append(res)

    # plot data for coincidences
    if PLOT_ALL_COINCIDENCE:
        plt.bar(time, coincidences, color=color_coincidence)
        plt.plot(time, res.best_fit,
                 color='k', ls='--')

        plt.title(f'Coincidences with {power} mW Pump Power')
        plt.xlabel('Time (ns)')
        plt.ylabel('Coincidences')
        plt.xlim((-5, 5))

        plt.tight_layout()
        plt.show()


# gather all fit data
all_amps = [res.params['amplitude'].value for res in all_res]
all_amps_err = [res.params['amplitude'].stderr for res in all_res]
all_kappa = []
all_kappa_err = []
all_g = []
all_g_err = []
all_integral = []
for res in all_res:
    # extract kappa data
    kappa = res.params['kappa'].value  # unit: 2*pi*GHz
    kappa /= 2*np.pi
    kappa *= 1e3  # unit: MHz
    kappa_err = res.params['kappa'].stderr
    kappa_err /= 2*np.pi
    kappa_err *= 1e3
    all_kappa.append(kappa)
    all_kappa_err.append(kappa_err)

    # extract g data
    g = res.params['g'].value  # unit: 2*pi*GHz
    g /= 2*np.pi
    g *= 1e3  # unit: MHz
    g_err = res.params['g'].stderr
    g_err /= 2*np.pi
    g_err *= 1e3
    all_g.append(g)
    all_g_err.append(g_err)

    # extract integral (numerically)
    only_coincidence = res.best_fit - res.params['amplitude'].value  # everything above bg
    integral = np.sum(only_coincidence[:-1])  # riemann sum
    integral /= integration_time  # convert to pairs/s
    integral /= (efficiency_48 * efficiency_52)  # convert to on-chip pair rate
    all_integral.append(integral)


# fit areas versus pump power
model = Model(quadratic)
res_area = model.fit(all_integral, x=powers,
                     a=0.16)
print(res_area.fit_report())


# plot of widths versus pump power
plt.errorbar(powers, all_kappa, yerr=all_kappa_err,
             color='coral', ls='', marker='o', capsize=3,
             label='Fitted Data')

plt.title('Coincidence Width versus Pump Power')
plt.xlabel('Pump Power (mW)')
plt.ylabel(r'Coincidence Fit $\kappa$ (MHz)')
plt.legend(shadow=True)

plt.tight_layout()
plt.show()


# plot of areas versus pump power
powers_for_fit = np.linspace(min(powers), max(powers), 100)
fig, ax = plt.subplots()

plt.errorbar(powers, all_integral,
             color='mediumpurple', ls='', marker='o', capsize=3,
             label='Fitted Data')
plt.plot(powers_for_fit, quadratic(powers_for_fit, **res_area.best_values),
         color='k', ls='--',
         label='Fitted Trend')

# add text
label = rf'$a$ = {res_area.params['a'].value:.3f} $\pm$ {res_area.params['a'].stderr:.3f}'
t = ax.text(0.95, 0.05, label,
            horizontalalignment='right', verticalalignment='bottom')
t.set_transform(ax.transAxes)

plt.title('Coincidence Area versus Pump Power')
plt.xlabel(r'Pump Power (mW)')
plt.ylabel(r'Coincidence Area (pairs/s)')
plt.legend(shadow=True)

plt.tight_layout()
plt.show()


# gather cavity data
cavity_df = pd.read_csv(CAVITY_DATA, header=10, skiprows=[11])
ramp = cavity_df['CH1'].astype(float)
transmission = cavity_df['CH2'].astype(float)

id_min = np.argmin(ramp)
id_max = np.argmax(ramp)
transmission = transmission[id_min:id_max]
transmission.reset_index(drop=True, inplace=True)
freq = np.linspace(0, SCAN_RANGE[1] - SCAN_RANGE[0],
                   num=(id_max-id_min))  # unit: MHz

# do fitting (and determine guesses for fit)
max_trans = max(transmission)
model = ConstantModel() + BreitWignerModel()
res = model.fit(transmission, x=freq,
                c=0.07,
                amplitude=0.1,
                center=2,
                sigma=0.1,
                q=0)
print("\nResonance fitting:")
print(res.fit_report())


# plot cavity fitting
plt.plot(freq, transmission,
         color='cornflowerblue', label='Data')
plt.plot(freq, res.best_fit,
         ls='--', color='k', label='Fit')

plt.title('Coincidence Pump Resonance')
plt.xlabel(f'Detuning from {SCAN_RANGE[0]:.3f} (GHz)')
plt.ylabel('Transmission (A.U.)')
plt.legend(shadow=True)

plt.tight_layout()
plt.show()


# get relevant info from cavity fit
width = res.params[f'sigma'].value  # unit: GHz
center = res.params[f'center'].value  # unit: GHz
amplitude = res.params[f'amplitude'].value
constant = res.params[f'c'].value

freq_light = SCAN_RANGE[0] + center  # unit: GHz
q = freq_light / width
contrast = amplitude / (amplitude + constant)
print("\n")
print(f"Cavity Q: {q}")
print(f"Cavity contrast: {contrast}")

power_enhance_1, power_enhance_2 = calculate_enhancement(freq_light*1e-3, q, contrast,
                                                         L=L, n_eff=n_eff)
gamma_1, gamma_2 = calculate_gamma(freq_light*1e-3, q, contrast,
                                   res_area.params['a'].value,
                                   L=L, n_eff=n_eff)

print("\n")
print(f"field enhancement (solution 1): {np.sqrt(power_enhance_1)}")
print(f"field enhancement (solution 2): {np.sqrt(power_enhance_2)}")
print("\n")
# print(f"gamma squared (solution 1): {gamma_1}")
# print(f"gamma squared (solution 2): {gamma_2}")
print(f"gamma (solution 1): {gamma_1}")
print(f"gamma (solution 2): {gamma_2}")

# calculate pair rate (for reference/double-checking)
power = 0.35
n_photons_1, n_photons_2 = calculate_rates(freq_light*1e-3, q, contrast,
                                           L=L, n_eff=n_eff, gamma=gamma_1, power=power)
n_photons_3, n_photons_4 = calculate_rates(freq_light*1e-3, q, contrast,
                                           L=L, n_eff=n_eff, gamma=gamma_2, power=power)

print("\n")
print(f"pair rate (solution 1): {n_photons_1}")
print(f"pair rate (solution 2): {n_photons_2}")
print(f"pair rate (solution 3): {n_photons_3}")
print(f"pair rate (solution 4): {n_photons_4}")

