import glob
import os
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel
from scipy.signal import find_peaks


DATA_DIR = ("/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators"
            "/Mounted_device_mk_3/300K/07112025")
OUTPUT_DIR = ("/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators"
              "/mounted_mk_3/300K_cavity/300K_07112025")
freq_start = 194756.163
freq_end = 194765.621

# fitting params
SMOOTHING = 5
PEAK_THRESH = 0.7

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')
color_smooth = 'coral'

PLOT_ALL_RES = True  # plot and save all intermediate results
PLOT_ALL_RES_TESTING = True


# moving average function for peaks
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# find and read oscilloscope files
filenames = glob.glob('dev*.csv', root_dir=DATA_DIR)
data_dfs = []
names = []
for file in filenames:
    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path, header=0, skiprows=[1])
    data_dfs.append(df)
    names.append(os.path.splitext(file)[0])


for name, data_df in zip(names, data_dfs):
    ramp = data_df['CH1'].astype(float)
    transmission = data_df['CH2'].astype(float)

    id_min = np.argmin(ramp)
    id_max = np.argmax(ramp)
    transmission = transmission[id_min:id_max]
    transmission.reset_index(drop=True, inplace=True)
    freq = np.linspace(0, (freq_end - freq_start) * 1e3,
                       num=(id_max - id_min))  # unit: MHz

    # find peaks to determine number of fits
    smoothed_data = moving_average(transmission.to_numpy(),
                                   n=SMOOTHING)
    peaks, peak_res = find_peaks(-smoothed_data,
                                 prominence=0.015, distance=50, width=5)
    peaks_to_keep = [p for p in peaks
                     if transmission[p] < (max(transmission) * PEAK_THRESH)]
    print(f"\t\tnumber of peaks: {len(peaks_to_keep)}")

    # do fitting (and determine guesses for fit)
    max_trans = max(transmission)
    model = LinearModel()
    model_kwargs = {}
    amplitudes = []
    for i, peak_idx in enumerate(peaks_to_keep):
        model += BreitWignerModel(prefix=f'p{i}_')
        amp_guess = max_trans - transmission[peak_idx]
        amplitudes.append(amp_guess)
        model_kwargs[f'p{i}_amplitude'] = amp_guess
        model_kwargs[f'p{i}_center'] = freq[peak_idx]
        model_kwargs[f'p{i}_sigma'] = 100
        model_kwargs[f'p{i}_q'] = 0
    model_kwargs['intercept'] = max_trans - sum(amplitudes)
    model_kwargs['slope'] = 0
    res = model.fit(transmission, x=freq,
                    **model_kwargs)

    text = ""
    for i in range(len(peaks_to_keep)):
        width = res.params[f'p{i}_sigma'].value  # unit: MHz
        center = res.params[f'p{i}_center'].value  # unit: MHz
        amplitude = res.params[f'p{i}_amplitude'].value
        # calculate value of bg at center
        slope = res.params['slope'].value
        intercept = res.params['intercept'].value
        constant = slope * center + intercept

        plt.axvline(center, ls=':', color='k')

        # get q
        freq_light = (freq_start * 1e3) + center  # unit: MHz
        q = freq_light / width

        if i != 0:
            text += "\n"
        text += rf"$\kappa_{i + 1}$: {width:.3f} MHz"
        text += "\n"
        text += f"$Q_{i + 1}$: {q:.3}"

    if PLOT_ALL_RES:
        if PLOT_ALL_RES_TESTING:
            plt.plot(freq, transmission,
                     color=color, label='Data')
            plt.plot(freq[:-(SMOOTHING-1)], smoothed_data,
                     color=color_smooth, label='Smoothed Data')
            plt.plot(freq, res.init_fit, 'r--',
                     label='Initial Fit')
            plt.plot(freq, res.best_fit, 'k--',
                     label='Fit')

            plt.gcf().text(0.95, 0.5, text,
                           ha='right', va='center', bbox=bbox)

            plt.title(name)
            plt.xlabel('Detuning (MHz)')
            plt.ylabel('Transmission (A.U.)')
            plt.legend()

            save_name = f"{name}.png"
            output_path = os.path.join(OUTPUT_DIR, 'testing')
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            plt.savefig(os.path.join(output_path, save_name))
            plt.clf()

        plt.plot(freq, transmission,
                 color=color, label='Data')
        plt.plot(freq, res.best_fit, 'k--',
                 label='Fit')

        plt.gcf().text(0.95, 0.5, text,
                       ha='right', va='center', bbox=bbox)

        device = name[3:5]
        scan = name[-1]
        plt.title(f'Device {device} scan #{scan}')
        plt.xlabel('Detuning (MHz)')
        plt.ylabel('Transmission (A.U.)')
        plt.legend()

        save_name = f"{name}.png"
        output_path = os.path.join(OUTPUT_DIR, 'resonances')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, save_name))
        plt.clf()
