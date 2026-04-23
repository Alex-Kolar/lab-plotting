import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.constants import c
from lmfit.models import BreitWignerModel, ConstantModel


# data params
DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Unmounted_device_mk_3/2026_04_15/cavity_scan/device_36')
CSV_PATH = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Unmounted_device_mk_3/2026_04_15/cavity_scan/device_36/resonance_freq_data.csv')
LASER_OFF_PATH = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
                  '/Unmounted_device_mk_3/2026_04_15/cavity_scan/device_36/LASEROFF.csv')
signal_file = 29
pump_file = 16
idler_file = 3

center_guess = 15
amp_guess = 2
c_guess = 0.5
q_guess = 0

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')
PLOT_WL = True  # if true, use wavelength for x-axis; if false, use frequency
PLOT_DB = False  # if true, use dB below ref for x-axis; if false, use linear transmission
db_ref_level = 2.5


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
num_files = len(main_df['File'])

# data processing and plotting
fig, ax = plt.subplots(figsize=(6, 3), dpi=300)

signal_freq = None
pump_freq = None
idler_freq = None
signal_trans = None
pump_trans = None
idler_trans = None

for _, row in main_df.iterrows():
    file_num = row['File'].astype(int)
    min_freq = row['Minimum (GHz)']
    max_freq = row['Maximum (GHz)']

    data_path = os.path.join(DATA_DIR, f'data_{file_num:06}.csv')
    data_df = pd.read_csv(data_path)

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

    # plot this row individually if requested
    if file_num == signal_file:
        signal_freq = freq
        signal_trans = transmission
    elif file_num == pump_file:
        pump_freq = freq
        pump_trans = transmission
    elif file_num == idler_file:
        idler_freq = freq
        idler_trans = transmission

    # plot transmission (as dB if requested)
    # NOTE: since files overlap significantly, only plot every other file
    if file_num % 2 == 0:
        if PLOT_DB:
            transmission = np.log10(transmission / db_ref_level) * 10
        if PLOT_WL:
            ax.plot(wl, transmission, color=color)
        else:
            ax.plot(freq, transmission, color=color)

ax.set_title('Source Resonance Scan')
if PLOT_WL:
    ax.set_xlabel('Wavelength (nm)')
    # ax.set_xlim(1536, 1539)
else:
    ax.set_xlabel('Frequency (THz)')
if PLOT_DB:
    ax.set_ylabel('Transmission (dB)')
    ax.set_ylim(-8, 0)
else:
    ax.set_ylabel('Transmission (A.U.)')
    ax.set_ylim(0, 2.5)
# ax.set_xlim(194.82, 194.84)

fig.tight_layout()
fig.show()


# fitting and plotting of selected files
labels = ['Signal', 'Pump', 'Idler']
for freq, trans, label in zip([signal_freq, pump_freq, idler_freq],
                              [signal_trans, pump_trans, idler_trans],
                              labels):
    print(f'Fitting {label} resonance')
    res, freq_detuning = resonance_helper(freq, trans, print_report=True)

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    ax.plot(freq_detuning, trans, color=color)
    ax.plot(freq_detuning, res.best_fit, '--k')
    ax.set_xlim(-2, 2)
    ax.set_title(f'{label} Resonance')
    ax.set_xlabel('Detuning from Cavity Center (GHz)')
    ax.set_ylabel('Transmission (A.U.)')

    fig.tight_layout()
    fig.show()
