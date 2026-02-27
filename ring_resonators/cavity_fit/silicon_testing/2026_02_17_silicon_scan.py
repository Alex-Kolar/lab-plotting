import os
import pandas as pd
import glob as glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from lmfit.models import BreitWignerModel, LinearModel
from scipy.signal import find_peaks
import pickle


DATA_DIR = ('/Users/alexkolar/Library/CloudStorage/Box-Box/Zhonglab/Lab data/Ring Resonators'
            '/Silicon_test_devices/mk_5/chip_2/2026_02_17/cavity_scan')
FREQ_FILE = 'resonance_freq_data.csv'
OUTPUT_DIR = ('/Users/alexkolar/Desktop/Lab/lab-plotting/output_figs/ring_resonators'
              '/silicon_testing/silicon_mk_5/chip_2_clad')

# fitting params
SMOOTHING = 21
PEAK_THRESH = 0.8
PEAK_DISTANCE = 1000

# plotting params
mpl.rcParams.update({'font.sans-serif': 'Helvetica',
                     'font.size': 12})
color = 'cornflowerblue'
color_smooth = 'coral'
bbox = dict(boxstyle='square', facecolor='white', alpha=1, edgecolor='black')
color_factor = 'black'
colormap = mpl.colormaps['Purples']
color_rate = 'coral'

PLOT_ALL_RES = True  # plot and save all intermediate results
PLOT_ALL_RES_TESTING = True  # plot and save all intermediate results with annotated fit information


# moving average function for peaks
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# find all device data
all_device_dirs = glob.glob('device_*', root_dir=DATA_DIR)
all_device_dirs = sorted(all_device_dirs)
all_devices = [int(device[-2:]) for device in all_device_dirs]

# save specific params for plotting combined results
center_by_device = {}
q_by_device = {}
contrast_by_device = {}

for device, device_dir in zip(all_devices, all_device_dirs):
    full_device_dir = os.path.join(DATA_DIR, device_dir)
    freq_data_df = pd.read_csv(os.path.join(full_device_dir, FREQ_FILE))

    if PLOT_ALL_RES_TESTING:
        output_path_test = os.path.join(OUTPUT_DIR, 'testing', f'{device}')
        if not os.path.exists(output_path_test):
            os.makedirs(output_path_test)
    if PLOT_ALL_RES:
        output_path = os.path.join(OUTPUT_DIR, 'resonances', f'{device}')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

    for row_idx, row in freq_data_df.iterrows():
        # get info
        freq_start = row['Minimum (GHz)']  # unit: GHz
        freq_end = row['Maximum (GHz)']  # unit: GHz
        file_no = int(row['File'])

        print(f'\tFitting row {row_idx + 1} for device {device}')

        # get oscilloscope data
        file_path = os.path.join(full_device_dir, f'data_{file_no:06d}.csv')
        data_df = pd.read_csv(file_path)

        ramp = data_df['Ramp Voltage (V)'].astype(float)
        transmission = data_df['Data Voltage (V)'].astype(float)
        # transmission -= bg_avg

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
                                     prominence=0.015, distance=PEAK_DISTANCE, width=10)
        peaks_to_keep = [p for p in peaks
                         if transmission[p] < (max(transmission) * PEAK_THRESH)]
        print(f"\t\tnumber of peaks: {len(peaks_to_keep)}")

        # plt.plot(smoothed_data)
        # plt.show()
        # pass

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
            model_kwargs[f'p{i}_sigma'] = 2000
            model_kwargs[f'p{i}_q'] = 0
        model_kwargs['intercept'] = max_trans - sum(amplitudes)
        model_kwargs['slope'] = 0
        res = model.fit(transmission, x=freq,
                        **model_kwargs)

        # extract data
        total_amp = sum([res.params[f'p{i}_amplitude'].value for i in range(len(peaks_to_keep))])
        wl = 299792458 / (freq_start * 1e9)  # unit: m
        wl *= 1e9  # unit: nm

        all_center_curr = []
        all_q_curr = []
        all_contrast_curr = []
        text = ""
        for i in range(len(peaks_to_keep)):
            width = res.params[f'p{i}_sigma'].value  # unit: MHz
            center = res.params[f'p{i}_center'].value  # unit: MHz
            amplitude = res.params[f'p{i}_amplitude'].value
            # calculate value of bg at center
            slope = res.params['slope'].value
            intercept = res.params['intercept'].value
            constant = slope * center + intercept

            # get q
            freq_light = (freq_start * 1e3) + center  # unit: MHz
            q = freq_light / width

            # calculate contrast
            contrast = amplitude / (total_amp + constant)

            # save data
            all_center_curr.append(freq_light)
            all_q_curr.append(q)
            all_contrast_curr.append(contrast)

            if i != 0:
                text += "\n"
            text += rf"$\kappa_{i + 1}$: {width:.3f} MHz"
            text += "\n"
            text += f"$Q_{i + 1}$: {q:.3}"

        print('\t\tqs:', all_q_curr)
        print('\t\tcontrasts:', all_contrast_curr)
        print('\t\tcenters:', all_center_curr)

        if device in center_by_device:
            center_by_device[device] += all_center_curr
            q_by_device[device] += all_q_curr
            contrast_by_device[device] += all_contrast_curr
        else:
            center_by_device[device] = all_center_curr
            q_by_device[device] = all_q_curr
            contrast_by_device[device] = all_contrast_curr

        if PLOT_ALL_RES:
            if PLOT_ALL_RES_TESTING:
                plt.plot(freq, transmission,
                         color=color, label='Data')
                plt.plot(freq[:-(SMOOTHING - 1)], smoothed_data,
                         color=color_smooth, label='Smoothed Data')
                plt.plot(freq, res.init_fit, 'r--',
                         label='Initial Fit')
                plt.plot(freq, res.best_fit, 'k--',
                         label='Fit')

                plt.gcf().text(0.95, 0.5, text,
                               ha='right', va='center', bbox=bbox)

                plt.title(f"Device {device} {wl:.3f} nm ({freq_start * 1e-3:.3f} THz) scan")
                plt.xlabel('Detuning (MHz)')
                plt.ylabel('Transmission (A.U.)')
                plt.legend()

                save_name = f"scan_{file_no}_wl_{wl:.3f}"
                save_name = save_name.replace(".", "_")
                save_name += ".png"

                plt.savefig(os.path.join(output_path_test, save_name))
                plt.clf()

            plt.plot(freq, transmission,
                     color=color, label='Data')
            plt.plot(freq, res.best_fit, 'k--',
                     label='Fit')

            plt.gcf().text(0.95, 0.5, text,
                           ha='right', va='center', bbox=bbox)

            plt.title(f"Device {device} {wl:.3f} nm ({freq_start * 1e-3:.3f} THz) scan")
            plt.xlabel('Detuning (MHz)')
            plt.ylabel('Transmission (A.U.)')
            plt.legend()

            save_name = f"scan_{file_no}_wl_{wl:.3f}"
            save_name = save_name.replace(".", "_")
            save_name += ".png"

            plt.savefig(os.path.join(output_path, save_name))
            plt.clf()

# save data for q
save_data = {"q": q_by_device,
             "centers": center_by_device,
             "contrasts": contrast_by_device}
save_name_data = f"res_data_2.bin"
with open(os.path.join(OUTPUT_DIR, save_name_data), "wb") as f:
    pickle.dump(save_data, f)

# xs = np.arange(len(all_device_dirs))
# plt.bar(xs, all_highest_q)
# plt.show()
#
# plt.bar(xs, all_highest_contrast)
# plt.show()
